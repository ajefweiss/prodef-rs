use crate::{Density, domain::Domain, sampling::RejectionSampling, tval};
use nalgebra::{Dim, OVector, RealField, SVector, Scalar, U1, VectorView};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal, StandardUniform};
use serde::{Deserialize, Serialize};

/// A lognormal PDF.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LognormalDensity<T>(T, T, Domain<T, U1>)
where
    T: Scalar;

impl<T> LognormalDensity<T>
where
    T: RealField,
{
    /// Create a new [`LognormalDensity`].
    ///
    /// Returns `None` if:
    /// - `sigma <= 0` (standard deviation must be positive)
    /// - `a >= b` (invalid bounds)
    /// - `a < 0` (lower bound must be non-negative, as lognormal is defined on (0, ∞))
    pub fn new(mu: T, sigma: T, lower_bound: T, upper_bound: T) -> Option<Self> {
        if sigma <= T::zero() {
            return None;
        }

        if lower_bound >= upper_bound || lower_bound < T::zero() {
            return None;
        }

        let domain = Domain::new_mdomain(OVector::from_element_generic(
            U1,
            U1,
            (Some(lower_bound), Some(upper_bound)),
        ));

        Some(Self(mu, sigma, domain))
    }

    /// Returns the maximum value of the domain.
    pub fn maximum(&self) -> T {
        match &self.2.inner().unwrap() {
            (_, Some(max)) => max.clone(),
            // Safe: LognormalDensity constructor validates a < b condition
            // and creates MDomain with explicit bounds, guaranteeing (min, max) both Some.
            _ => unreachable!("MDomain always has explicit bounds in LognormalDensity"),
        }
    }

    /// Returns the minimum value of the domain.
    pub fn minimum(&self) -> T {
        match &self.2.inner().unwrap() {
            (Some(min), _) => min.clone(),
            // Safe: LognormalDensity constructor validates a < b condition
            // and creates MDomain with explicit bounds, guaranteeing (min, max) both Some.
            _ => unreachable!("MDomain always has explicit bounds in LognormalDensity"),
        }
    }

    /// Returns the mean of the underlying normal distribution (μ parameter).
    pub fn mu(&self) -> T {
        self.0.clone()
    }

    /// Returns the standard deviation of the underlying normal distribution (σ parameter).
    pub fn sigma(&self) -> T {
        self.1.clone()
    }
}

impl<T> Density<T, U1> for LognormalDensity<T>
where
    T: RealField,
    StandardNormal: Distribution<T>,
    StandardUniform: Distribution<T>,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, U1, RStride, CStride>,
    ) -> Option<T> {
        if !self.2.contains(sample) {
            return None;
        }

        let x = sample[0].clone();

        // f(x) = 1 / (x * σ * √(2π)) * exp(-(ln(x) - μ)² / (2σ²))
        let ln_x = x.clone().ln();
        let z = (ln_x - self.0.clone()) / self.1.clone();

        Some(
            T::one() / (x * self.1.clone() * tval!(2.0 * std::f64::consts::PI, f64).sqrt())
                * (-z.powi(2) / tval!(2, usize)).exp(),
        )
    }

    fn domain(&self) -> Domain<T, U1> {
        self.2.clone()
    }

    fn mean(&self) -> SVector<T, 1> {
        SVector::from([(self.0.clone() + self.1.clone().powi(2) / tval!(2, usize)).exp()])
    }

    fn sample<R>(&self, rng: &mut R) -> Option<SVector<T, 1>>
    where
        R: RngExt + SeedableRng,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<U1>,
    {
        self.rejection_sample(rng)
    }

    fn sample_iter<R>(&self, rng: &mut R) -> impl Iterator<Item = Option<SVector<T, 1>>>
    where
        R: RngExt + SeedableRng,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<U1>,
    {
        let normal = StandardNormal;
        let mu = self.0.clone();
        let sigma = self.1.clone();
        let domain = self.2.clone();

        rng.sample_iter(normal).map(move |value| {
            // If z ~ N(0, 1), then exp(μ + σ*z) ~ LogNormal(μ, σ)
            let sample =
                OVector::from_element_generic(U1, U1, (mu.clone() + sigma.clone() * value).exp());

            // Check if sample is within domain bounds
            if domain.contains::<U1, U1>(&sample.as_view()) {
                Some(sample)
            } else {
                None
            }
        })
    }

    fn variance(&self) -> SVector<T, 1> {
        let sigma_sq = self.1.clone().powi(2);

        let f1 = sigma_sq.clone().exp() - T::one();
        let f2 = (tval!(2, usize) * self.0.clone() + sigma_sq).exp();

        SVector::from([(f1 * f2)])
    }
}

impl<T> RejectionSampling<T, U1> for LognormalDensity<T>
where
    T: RealField,
    StandardNormal: Distribution<T>,
    StandardUniform: Distribution<T>,
{
    fn rejection_candidate<R>(&self, rng: &mut R) -> (SVector<T, 1>, Option<T>)
    where
        R: RngExt + SeedableRng,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<U1>,
    {
        let z = rng.sample(StandardNormal);

        let candidate = SVector::from([(self.0.clone() + self.1.clone() * z).exp()]);

        (candidate, None)
    }

    fn scale_factor(&self) -> T {
        T::one()
    }
}

impl<T: RealField> TryFrom<crate::univariate::UnivariateDensity<T>> for LognormalDensity<T> {
    type Error = ();

    fn try_from(value: crate::univariate::UnivariateDensity<T>) -> Result<Self, Self::Error> {
        match value {
            crate::univariate::UnivariateDensity::Lognormal(pdf) => Ok(pdf),
            _ => Err(()),
        }
    }
}
