use crate::{
    Density, Domain, RejectionSampler, SamplingMode, macros::tval,
    multinormal::MultivariateNormalDensity, particle::ParticleDensity,
};
use nalgebra::{
    DVector, DefaultAllocator, Dim, Dyn, MatrixView, OMatrix, OVector, RealField, U1, VectorView,
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
    /// Create a [`ParticleDensity`] from a view of vectors, with optional weights and kernel.
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

    /// Create a [`ParticleDensity`] from a owned matrix of vectors, with optional weights and kernel.
    ///
    /// The kernel is used for sampling only, and if no kernel is provided, a default kernel will
    /// be constructed using the covariance of the particle population.
    pub fn new(
        matrix: OMatrix<T, D, Dyn>,
        domain: Domain<T, D>,
        opt_weights: Option<&[T]>,
        opt_kernel: Option<MultivariateNormalDensity<T, D>>,
    ) -> Option<Self>
    where
        T: Sum,
    {
        Self::from_vectors(&matrix.as_view(), domain, opt_weights, opt_kernel)
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

                                if delta.norm() == T::zero() {
                                    T::zero()
                                } else {
                                    ((weight_old.clone()).ln() - self.kernel.mahalanobis_distance_sq(&delta.as_view())).exp()
                                }
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

impl<T, D> Density<T, D> for ParticleDensity<T, D, MultivariateNormalDensity<T, D>>
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

    fn mean(&self) -> OVector<T, D> {
        // Compute the weighted mean of particles
        match &self.opt_weights {
            Some(weights) => OVector::from_iterator_generic(
                self.particles.shape_generic().0,
                U1,
                (0..self.particles.nrows()).map(|i| {
                    self.particles
                        .row(i)
                        .iter()
                        .zip(weights.iter())
                        .map(|(p, w)| p.clone() * w.clone())
                        .sum::<T>()
                }),
            ),
            None => {
                let count = tval!(self.particles.ncols(), usize);
                OVector::from_iterator_generic(
                    self.particles.shape_generic().0,
                    U1,
                    (0..self.particles.nrows())
                        .map(|i| self.particles.row(i).iter().cloned().sum::<T>() / count.clone()),
                )
            }
        }
    }

    fn sample(&self, rng: &mut impl RngExt, mode: &SamplingMode) -> Option<OVector<T, D>> {
        self.rejection_sample(rng, mode)
    }

    fn sample_iter(&self, rng: &mut impl RngExt) -> impl Iterator<Item = Option<OVector<T, D>>> {
        let particle = self.sample_particle(rng);

        repeat_with(move || {
            let candidate = &particle
                + self
                    .kernel
                    .sample(rng, &SamplingMode::SingleAttempt)
                    .expect("particle kernel should use an unbounded domain")
                - &self.kernel.mean;

            // Check if sample is within domain bounds
            if self.domain.contains(&candidate.as_view()) {
                Some(candidate)
            } else {
                None
            }
        })
    }

    fn variance(&self) -> OVector<T, D> {
        let mean = self.mean();

        // Compute weighted variance for each dimension
        match &self.opt_weights {
            Some(weights) => OVector::from_iterator_generic(
                self.particles.shape_generic().0,
                U1,
                (0..self.particles.nrows()).map(|i| {
                    self.particles
                        .row(i)
                        .iter()
                        .zip(weights.iter())
                        .map(|(p, w)| {
                            let diff = p.clone() - mean[i].clone();
                            w.clone() * diff.clone() * diff
                        })
                        .sum::<T>()
                }),
            ),
            None => {
                let count = tval!(self.particles.ncols(), usize);
                OVector::from_iterator_generic(
                    self.particles.shape_generic().0,
                    U1,
                    (0..self.particles.nrows()).map(|i| {
                        self.particles
                            .row(i)
                            .iter()
                            .map(|p| {
                                let diff = p.clone() - mean[i].clone();
                                diff.clone() * diff
                            })
                            .sum::<T>()
                            / count.clone()
                    }),
                )
            }
        }
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
            + self
                .kernel
                .sample(rng, &SamplingMode::UntilValidNoLimit)
                .expect("particle kernel should use an unbounded domain")
            - &self.kernel.mean
    }
}
