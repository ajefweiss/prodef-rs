//! A module that implements the Kent (5-parameter Fisher-Bingham) distribution.
use crate::{Density, Domain, sampling::MetropolisHastingsSampling};
use approx::ulps_eq;
use nalgebra::{
    Const, DefaultAllocator, Dim, OVector, RealField, Vector3, VectorView, allocator::Allocator,
};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};

/// A struct that represents the Kent distribution.
#[derive(Clone, Debug)]
pub struct KentDensity<T> {
    mu: Vector3<T>,
    g1: Vector3<T>,
    kappa: T,
    beta: T,
}

impl<T> KentDensity<T>
where
    T: RealField,
{
    /// Creates a new Kent distribution with the given parameters.
    pub fn new(mu: Vector3<T>, g1: Vector3<T>, kappa: T, beta: T) -> Self {
        Self {
            mu: mu.clone().normalize(),
            g1,
            kappa,
            beta,
        }
    }
}

impl<T> Density<T, Const<3>> for KentDensity<T>
where
    T: RealField + SampleUniform,
    StandardNormal: Distribution<T>,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, Const<3>, RStride, CStride>,
    ) -> Option<T> {
        self.log_density(sample).map(|ld| ld.exp())
    }

    fn log_density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, Const<3>, RStride, CStride>,
    ) -> Option<T> {
        let norm = sample.norm();

        // Check if sample is on the unit sphere
        if ulps_eq!(norm.clone(), T::one()) {
            return None;
        }

        // Compute the dot product between sample and mean direction mu
        let mu_dot_sample = self.mu.dot(&sample);

        // Compute g2 as the normalized second major axis (perpendicular to both mu and g1)
        let g2 = self.mu.cross(&self.g1).normalize();

        // Compute projections onto g1 and g2
        let g1_dot_sample = self.g1.dot(&sample);
        let g2_dot_sample = g2.dot(&sample);

        // Kent distribution log-density:
        // log f(x) = kappa * (mu · x) + beta * (g1 · x)^2 - (g2 · x)^2
        let log_density = self.kappa.clone() * mu_dot_sample
            + self.beta.clone() * (g1_dot_sample.powi(2) - g2_dot_sample.powi(2));

        Some(log_density)
    }

    fn domain(&self) -> Domain<T, Const<3>>
    where
        DefaultAllocator: Allocator<Const<3>>,
    {
        Domain::new_sdomain(Const::<3>)
    }

    fn mean(&self) -> OVector<T, Const<3>> {
        self.mu.clone_owned()
    }

    fn sample<R>(&self, rng: &mut R) -> Option<OVector<T, Const<3>>>
    where
        R: RngExt + SeedableRng,
        DefaultAllocator: Allocator<Const<3>>,
    {
        Some(self.mh_sample(rng))
    }

    fn sample_iter<R>(&self, rng: &mut R) -> impl Iterator<Item = Option<OVector<T, Const<3>>>>
    where
        R: RngExt + SeedableRng,
        DefaultAllocator: Allocator<Const<3>>,
    {
        self.mh_sample_iter(rng).map(Some)
    }

    fn variance(&self) -> OVector<T, Const<3>>
    where
        DefaultAllocator: Allocator<Const<3>>,
    {
        // Variance is not straightforward for directional distributions
        // Return a placeholder
        Vector3::new(T::one(), T::one(), T::one())
    }
}

impl<T> MetropolisHastingsSampling<T, Const<3>> for KentDensity<T>
where
    T: RealField + SampleUniform,
    StandardNormal: Distribution<T>,
    DefaultAllocator: Allocator<Const<3>>,
{
    fn mh_candidate<R>(&self, current: OVector<T, Const<3>>, rng: &mut R) -> OVector<T, Const<3>>
    where
        R: RngExt + SeedableRng,
    {
        // Proposal: small random perturbation on the unit sphere
        // Generate a random tangent vector and add to current point
        let z1 = rng.sample(StandardNormal);
        let z2 = rng.sample(StandardNormal);
        let z3 = rng.sample(StandardNormal);

        let proposal_step = Vector3::new(z1, z2, z3) * T::from_f64(0.1).unwrap();
        let candidate = (current + proposal_step).normalize();

        candidate
    }

    fn mh_initialze(&self) -> OVector<T, Const<3>>
    where
        DefaultAllocator: Allocator<Const<3>>,
    {
        self.mu.normalize()
    }
}
