use crate::{
    Density, Domain, RejectionSampler, SamplingMode, macros::tval,
    multinormal::MultivariateNormalDensity, particle::ParticleDensity,
};
use nalgebra::{
    DVector, DefaultAllocator, Dim, Dyn, MatrixView, OVector, RealField, U1, VectorView,
    allocator::Allocator,
};
use rand::RngExt;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use rayon::prelude::*;
use std::iter::{Sum, repeat_with};

impl<T, D> ParticleDensity<T, D, MultivariateNormalDensity<T, D>>
where
    T: RealField + Sum,
    D: Dim,
    StandardNormal: Distribution<T>,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D> + Allocator<D, Dyn>,
{
    /// Create a [`ParticleDensity`] from a set of vectors, with optional weights and kernel.
    ///
    /// The kernel is used for sampling only, and if no kernel is provided, a default kernel will
    /// be constructed using the covariance of the particle population.
    pub fn from_vectors<RStride: Dim, CStride: Dim>(
        view: &MatrixView<T, D, Dyn, RStride, CStride>,
        domain: Domain<T, D>,
        opt_weights: Option<&[T]>,
        opt_kernel: Option<MultivariateNormalDensity<T, D>>,
    ) -> Option<Self>
    where
        T: Sum,
    {
        let n_dim = view.shape_generic().0;
        let n_dim_b = domain.shape_generic();

        // Check dimensions (only required for D = Dyn).
        if n_dim.value() != n_dim_b.value() {
            return None;
        }

        let kernel: MultivariateNormalDensity<T, D> = match opt_kernel {
            Some(k) => k,
            None => {
                let mut mvnk = MultivariateNormalDensity::from_vectors(
                    view,
                    Domain::new_udomain(n_dim),
                    opt_weights,
                )?;

                mvnk.mean = OVector::from_iterator_generic(
                    n_dim,
                    U1,
                    (0..n_dim.value()).map(|_| T::zero()),
                );

                mvnk
            }
        };

        let count = view.ncols();

        Some(Self {
            particles: view.clone_owned(),
            opt_weights: opt_weights.map(|w| DVector::from_iterator(count, w.iter().cloned())),
            domain,
            kernel,
        })
    }

    /// Compute non-normalized transition weights for a new particle matrix.
    pub fn transition_weights<RStride: Dim, CStride: Dim>(
        &self,
        new_particles: &MatrixView<T, D, Dyn, RStride, CStride>,
    ) -> Vec<T>
    where
        <DefaultAllocator as Allocator<D>>::Buffer<T>: Sync,
        <DefaultAllocator as Allocator<D, D>>::Buffer<T>: Sync,
        <DefaultAllocator as Allocator<D, Dyn>>::Buffer<T>: Sync,
        <DefaultAllocator as Allocator<D>>::Buffer<(Option<T>, Option<T>)>: Sync,
    {
        match self.opt_weights {
            Some(ref weights) => new_particles
                .par_column_iter()
                .map(|params| {
                    T::one()
                        / self
                            .particles
                            .column_iter()
                            .zip(weights.iter())
                            .map(|(params_old, weight_old)| {
                                let delta = params.clone() - params_old;

                                (weight_old.clone().ln()
                                    - self.kernel.mahalanobis_distance_sq(&delta.as_view()))
                                .exp()
                            })
                            .sum::<T>()
                })
                .collect::<Vec<T>>(),
            None => new_particles
                .par_column_iter()
                .map(|params| {
                    T::one()
                        / self
                            .particles
                            .column_iter()
                            .map(|params_old| {
                                let delta = params.clone() - params_old;

                                (-self.kernel.mahalanobis_distance_sq(&delta.as_view())).exp()
                            })
                            .sum::<T>()
                })
                .collect::<Vec<T>>(),
        }
    }
}

impl<T, D> Density<T, D> for &ParticleDensity<T, D, MultivariateNormalDensity<T, D>>
where
    T: RealField + SampleUniform + Sum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D> + Allocator<D, Dyn>,
    StandardNormal: Distribution<T>,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, D, RStride, CStride>,
    ) -> Option<T> {
        if !self.domain.contains(sample) {
            return None;
        }

        let count = tval!(self.count(), usize);
        let rank = tval!(self.kernel.rank(), usize);

        // Construct the bandwidth matrix K_H using Silverman's rule of thumb: K_H = K * (1/n)^(1/(d+4))
        let bandwidth = self.kernel.clone()
            * (T::one() / count.powf(T::one() / (rank + tval!(4, usize)))).powi(2);

        let normalization = bandwidth.normalization_factor();

        Some(
            normalization / tval!(self.count(), usize)
                * match &self.opt_weights {
                    Some(weights) => self
                        .particles
                        .column_iter()
                        .zip(weights.iter())
                        .map(|(col, weight)| {
                            let delta_x = sample - col;

                            weight.clone()
                                * (-bandwidth.mahalanobis_distance_sq(&delta_x.as_view())
                                    / tval!(2, usize))
                                .exp()
                        })
                        .sum::<T>(),

                    None => self
                        .particles
                        .column_iter()
                        .map(|col| {
                            let delta_x = sample - col;

                            (-bandwidth.mahalanobis_distance_sq(&delta_x.as_view())
                                / tval!(2, usize))
                            .exp()
                        })
                        .sum::<T>(),
                },
        )
    }

    fn domain(&self) -> Domain<T, D> {
        self.domain.clone()
    }

    fn sample(&self, rng: &mut impl RngExt, mode: &SamplingMode) -> Option<OVector<T, D>> {
        self.rejection_sample(rng, mode)
    }

    fn sample_iter(&self, rng: &mut impl RngExt) -> impl Iterator<Item = Option<OVector<T, D>>> {
        let particle = self.sample_particle(rng);

        repeat_with(move || {
            let candidate = &particle
                + (&self.kernel)
                    .sample(rng, &SamplingMode::SingleAttempt)
                    .expect("particle kernel should use an unbounded domain")
                - &self.kernel.mean;

            if self.domain.contains(&candidate.as_view()) {
                Some(candidate)
            } else {
                None
            }
        })
    }
}

impl<T, D> RejectionSampler<T, D> for &ParticleDensity<T, D, MultivariateNormalDensity<T, D>>
where
    T: RealField + SampleUniform + Sum,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D> + Allocator<D, Dyn>,
    StandardNormal: Distribution<T>,
{
    fn generate_candidate(&self, rng: &mut impl RngExt) -> OVector<T, D> {
        let particle = self.sample_particle(rng);

        &particle
            + (&self.kernel)
                .sample(rng, &SamplingMode::UntilValidNoLimit)
                .expect("particle kernel should use an unbounded domain")
            - &self.kernel.mean
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Domain;
    use approx::ulps_eq;
    use nalgebra::{Matrix, SVector, U2, U3, VecStorage};
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_particle_density() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let uniform = StandardNormal;

        let array_0 = Matrix::<f64, U2, Dyn, VecStorage<f64, U2, Dyn>>::from_iterator(
            10000,
            (0..20000).map(|idx| {
                if idx % 2 == 0 {
                    0.1 + rng.sample::<f64, StandardNormal>(uniform)
                } else {
                    0.25 + rng.sample::<f64, StandardNormal>(uniform)
                }
            }),
        );

        let mvpdf_0 = MultivariateNormalDensity::from_vectors::<Dyn, U2>(
            &array_0.as_view(),
            Domain::new_mdomain(SVector::from([
                (Some(-0.75), Some(0.75)),
                (Some(-0.75), Some(0.75)),
            ])),
            None,
        )
        .unwrap();

        let ptpdf_0 = ParticleDensity::from_vectors::<U1, U2>(
            &array_0.as_view(),
            Domain::new_mdomain(SVector::from([
                (Some(-0.75), Some(0.75)),
                (Some(-0.75), Some(0.75)),
            ])),
            None,
            None,
        )
        .unwrap();

        assert!(ulps_eq!(
            (&mvpdf_0)
                .density::<U1, U2>(&SVector::from([0.2, 0.35]).as_view())
                .unwrap(),
            0.161284,
            epsilon = 1e-5,
            max_ulps = 5
        ));

        assert!(ulps_eq!(
            (&ptpdf_0)
                .density::<U1, U2>(&SVector::from([0.2, 0.35]).as_view())
                .unwrap(),
            0.155155,
            epsilon = 1e-5,
            max_ulps = 5
        ));

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);

        let array = Matrix::<f64, U3, Dyn, VecStorage<f64, U3, Dyn>>::from_iterator(
            10000,
            (0..30000).map(|idx| {
                if idx % 3 == 0 {
                    0.1 + rng.sample::<f64, StandardNormal>(uniform)
                } else if idx % 3 == 1 {
                    0.0
                } else {
                    0.25 + rng.sample::<f64, StandardNormal>(uniform)
                }
            }),
        );

        let ptpdf = ParticleDensity::from_vectors::<U1, U3>(
            &array.as_view(),
            Domain::new_mdomain(SVector::from([
                (Some(-0.75), Some(0.75)),
                (Some(-0.75), Some(0.75)),
                (Some(-0.75), Some(0.75)),
            ])),
            None,
            None,
        )
        .unwrap();

        assert!(
            (&ptpdf)
                .density::<U1, U3>(&SVector::from([0.2, -0.85, 0.35]).as_view())
                .is_none()
        );

        assert!(ulps_eq!(
            (&ptpdf)
                .density::<U1, U3>(&SVector::from([0.2, 0.0, 0.35]).as_view())
                .unwrap(),
            0.155147,
            epsilon = 1e-5,
            max_ulps = 5
        ));

        assert!(ulps_eq!(
            (&ptpdf)
                .sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 100 })
                .unwrap(),
            SVector::from([-0.5195214192763392, 0.0, -0.4065653428511528,]),
            epsilon = 1e-5,
            max_ulps = 5
        ));
    }

    #[test]
    fn test_particle_count() {
        // Test that particle count is preserved
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let uniform = StandardNormal;

        let n_particles = 1000;
        let array = Matrix::<f64, U2, Dyn, VecStorage<f64, U2, Dyn>>::from_iterator(
            n_particles,
            (0..n_particles * 2).map(|idx| {
                if idx % 2 == 0 {
                    0.1 + rng.sample::<f64, StandardNormal>(uniform)
                } else {
                    0.25 + rng.sample::<f64, StandardNormal>(uniform)
                }
            }),
        );

        let ptpdf = ParticleDensity::from_vectors::<U1, U2>(
            &array.as_view(),
            Domain::new_udomain(U2),
            None,
            None,
        )
        .unwrap();

        // Check that particle count matches
        assert_eq!(ptpdf.particles.ncols(), n_particles);
    }

    #[test]
    fn test_particle_weighted_sampling() {
        // Test particle density with explicit weights
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let uniform = StandardNormal;

        let n_particles = 100;
        let array = Matrix::<f64, U1, Dyn, VecStorage<f64, U1, Dyn>>::from_iterator(
            n_particles,
            (0..n_particles).map(|_| rng.sample::<f64, StandardNormal>(uniform)),
        );

        // Create weights favoring first particles
        let weights = DVector::from_iterator(
            n_particles,
            (0..n_particles).map(|i| if i < n_particles / 2 { 2.0 } else { 1.0 }),
        );

        let ptpdf = ParticleDensity::from_vectors::<U1, U1>(
            &array.as_view(),
            Domain::new_udomain(U1),
            Some(weights.as_slice()),
            None,
        )
        .unwrap();

        // Test that samples can be drawn
        let mut samples = Vec::new();
        for _ in 0..50 {
            if let Some(sample) = (&ptpdf).sample(&mut rng, &SamplingMode::SingleAttempt) {
                samples.push(sample[0]);
            }
        }

        // Ensure some samples were successfully drawn
        assert!(!samples.is_empty());
    }

    #[test]
    fn test_particle_domain_enforcement() {
        // Test that particles outside domain return None
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let uniform = StandardNormal;

        let array = Matrix::<f64, U2, Dyn, VecStorage<f64, U2, Dyn>>::from_iterator(
            100,
            (0..200).map(|idx| {
                if idx % 2 == 0 {
                    0.1 + rng.sample::<f64, StandardNormal>(uniform)
                } else {
                    0.25 + rng.sample::<f64, StandardNormal>(uniform)
                }
            }),
        );

        let ptpdf = ParticleDensity::from_vectors::<U1, U2>(
            &array.as_view(),
            Domain::new_mdomain(SVector::from([
                (Some(0.0), Some(0.5)),
                (Some(0.0), Some(0.5)),
            ])),
            None,
            None,
        )
        .unwrap();

        // Test point outside domain
        let outside_sample = SVector::from([0.6, 0.3]);
        assert!(
            (&ptpdf)
                .density::<U1, U2>(&outside_sample.as_view())
                .is_none()
        );

        // Test point inside domain
        let inside_sample = SVector::from([0.3, 0.3]);
        assert!(
            (&ptpdf)
                .density::<U1, U2>(&inside_sample.as_view())
                .is_some()
        );
    }

    #[test]
    fn test_particle_statistical_validation_weighted_samples() {
        // Test that particle sampling generates valid samples
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        // Create two clusters of particles
        let array = Matrix::<f64, U1, Dyn, VecStorage<f64, U1, Dyn>>::from_iterator(
            100,
            (0..100).map(|i| if i < 50 { -1.0 } else { 1.0 }),
        );

        // Give more weight to positive cluster
        let weights: Vec<f64> = (0..100).map(|i| if i < 50 { 1.0 } else { 3.0 }).collect();

        let ptpdf = ParticleDensity::from_vectors::<U1, U1>(
            &array.as_view(),
            Domain::new_udomain(U1),
            Some(&weights),
            None,
        )
        .unwrap();

        // Generate samples and verify they're produced
        let mut sample_count = 0;

        for _ in 0..500 {
            if let Some(_sample) =
                (&ptpdf).sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
            {
                sample_count += 1;
            }
        }

        // Verify we can successfully sample from weighted particle distribution
        assert!(
            sample_count > 100,
            "Should generate many samples from weighted particle distribution, got {}",
            sample_count
        );
    }

    #[test]
    fn test_particle_statistical_validation_kernel_smoothness() {
        // Test that particle density evaluates correctly at particle locations
        // Create particles at two locations
        let array = Matrix::<f64, U2, Dyn, VecStorage<f64, U2, Dyn>>::from_iterator(
            2,
            vec![0.0, 0.0, 2.0, 2.0],
        );

        let ptpdf = ParticleDensity::from_vectors::<U1, U2>(
            &array.as_view(),
            Domain::new_udomain(U2),
            None,
            None,
        )
        .unwrap();

        // Test that density is non-zero at multiple points
        let density_at_particle = (&ptpdf)
            .density::<U1, U2>(&SVector::from([0.0, 0.0]).as_view())
            .unwrap();

        let density_at_second_particle = (&ptpdf)
            .density::<U1, U2>(&SVector::from([2.0, 2.0]).as_view())
            .unwrap();

        let density_at_midpoint = (&ptpdf)
            .density::<U1, U2>(&SVector::from([1.0, 1.0]).as_view())
            .unwrap();

        // All should be positive
        assert!(
            density_at_particle > 0.0,
            "Density at first particle should be positive, got {}",
            density_at_particle
        );
        assert!(
            density_at_second_particle > 0.0,
            "Density at second particle should be positive, got {}",
            density_at_second_particle
        );
        assert!(
            density_at_midpoint > 0.0,
            "Density at midpoint should be positive, got {}",
            density_at_midpoint
        );
    }

    #[test]
    fn test_particle_statistical_validation_ensemble_statistics() {
        // Test that particle ensemble sampling preserves ensemble statistics
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(44);
        let uniform = StandardNormal;

        // Create ensemble with known mean
        let ensemble_mean = 0.5;
        let n_particles = 500;
        let array = Matrix::<f64, U1, Dyn, VecStorage<f64, U1, Dyn>>::from_iterator(
            n_particles,
            (0..n_particles)
                .map(|_| ensemble_mean + rng.sample::<f64, StandardNormal>(uniform) * 0.1),
        );

        let ptpdf = ParticleDensity::from_vectors::<U1, U1>(
            &array.as_view(),
            Domain::new_udomain(U1),
            None,
            None,
        )
        .unwrap();

        // Sample from particle density
        let samples: Vec<f64> = (0..5000)
            .filter_map(|_| {
                (&ptpdf)
                    .sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
                    .map(|s| s[0])
            })
            .collect();

        assert!(!samples.is_empty(), "Should generate samples");

        let sample_mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;

        // Sample mean should be close to ensemble mean
        assert!(
            (sample_mean - ensemble_mean).abs() < 0.1,
            "Sample mean ({}) should be close to ensemble mean ({})",
            sample_mean,
            ensemble_mean
        );

        // Estimate variance from samples
        let sample_variance: f64 = samples
            .iter()
            .map(|x| (x - sample_mean).powi(2))
            .sum::<f64>()
            / samples.len() as f64;

        // Variance should be reasonable (particles have std 0.1, kernel adds some)
        assert!(
            sample_variance > 0.01 && sample_variance < 0.3,
            "Sample variance ({}) should be reasonable for small particle std",
            sample_variance
        );
    }
}
