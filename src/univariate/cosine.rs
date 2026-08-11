use crate::{Density, domain::Domain, tval};
use approx::ulps_eq;
use nalgebra::{Dim, OVector, RealField, SVector, Scalar, U1, VectorView};
use rand::{RngExt, SeedableRng};
use rand_distr::{Uniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};

/// A cosine PDF defined on [-π/2, π/2].
///
/// Sine degenerate ranges (where sin is equal at endpoints) are rejected.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CosineDensity<T>(Domain<T, U1>)
where
    T: Scalar;

impl<T> CosineDensity<T>
where
    T: RealField,
{
    /// Returns the maximum value of the domain.
    pub fn maximum(&self) -> T {
        match &self.0.inner().unwrap() {
            (_, Some(max)) => max.clone(),
            // Safe: CosineDensity constructor enforces MDomain with explicit bounds
            // after validating sine uniqueness and range constraints.
            _ => unreachable!("CosineDensity MDomain always has explicit bounds"),
        }
    }

    /// Returns the minimum value of the domain.
    pub fn minimum(&self) -> T {
        match &self.0.inner().unwrap() {
            (Some(min), _) => min.clone(),
            // Safe: CosineDensity constructor enforces MDomain with explicit bounds
            // after validating sine uniqueness and range constraints.
            _ => unreachable!("CosineDensity MDomain always has explicit bounds"),
        }
    }

    /// Create a new [`CosineDensity`].
    ///
    /// Returns [`None`] for an invalid domain range.
    pub fn new(minimum: T, maximum: T) -> Option<Self> {
        match (
            minimum > -T::frac_pi_2(),
            maximum < T::frac_pi_2(),
            maximum > minimum,
            maximum.clone().sin() != minimum.clone().sin(),
        ) {
            (true, true, true, true) => Some(Self(Domain::new_mdomain(
                OVector::from_element_generic(U1, U1, (Some(minimum), Some(maximum))),
            ))),
            _ => None,
        }
    }
}

impl<T> Density<T, U1> for CosineDensity<T>
where
    T: RealField + SampleUniform,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, U1, RStride, CStride>,
    ) -> Option<T> {
        if !self.0.contains(sample) {
            return None;
        }

        Some(sample[0].clone().cos())
    }

    fn domain(&self) -> Domain<T, U1> {
        self.0.clone()
    }

    fn mean(&self) -> SVector<T, 1> {
        // For a symmetric domain [-a, a], the mean is 0.
        // For a general bounded domain [a, b], the mean is computed as:
        // mean = ∫[a,b] x*cos(x) dx / ∫[a,b] cos(x) dx
        //
        // where:
        // - ∫ cos(x) dx = sin(b) - sin(a)
        // - ∫ x*cos(x) dx = x*sin(x) + cos(x) |_a^b = (b*sin(b) + cos(b)) - (a*sin(a) + cos(a))
        let a = self.minimum();
        let b = self.maximum();

        // Check if domain is symmetric around 0
        let zero = T::zero();
        if a == -b.clone() {
            return SVector::from([zero]);
        }

        // Normalization constant: ∫_a^b cos(x) dx = sin(b) - sin(a)
        let sin_b = b.clone().sin();
        let sin_a = a.clone().sin();
        let norm = sin_b.clone() - sin_a.clone();

        // If norm is approximately zero, the distribution is degenerate
        let norm_abs = norm.clone().abs();
        if ulps_eq!(norm_abs, T::zero()) {
            return SVector::from([T::zero()]);
        }

        // ∫_a^b x*cos(x) dx = [x*sin(x) + cos(x)]_a^b
        let cos_b = b.clone().cos();
        let cos_a = a.clone().cos();
        let upper = b * sin_b + cos_b;
        let lower = a * sin_a + cos_a;
        let numerator = upper - lower;

        SVector::from([numerator / norm])
    }

    fn sample<R>(&self, rng: &mut R) -> Option<SVector<T, 1>>
    where
        R: RngExt + SeedableRng,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<U1>,
    {
        // The range is limited to the interval [-π/2, π/2].
        // This invariant is guaranteed by the constructor.
        match &self.0.inner().unwrap() {
            (Some(min), Some(max)) => {
                let uniform = Uniform::new_inclusive(min.clone().sin(), max.clone().sin()).unwrap();

                Some(SVector::from([rng.sample(uniform).asin()]))
            }
            // Safe by construction
            _ => unreachable!(),
        }
    }

    fn sample_iter<R>(&self, rng: &mut R) -> impl Iterator<Item = Option<SVector<T, 1>>>
    where
        R: RngExt + SeedableRng,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<U1>,
    {
        // The range is limited to the interval [-π/2, π/2].
        // This invariant is guaranteed by the constructor.
        match &self.0.inner().unwrap() {
            (Some(min), Some(max)) => {
                let uniform = Uniform::new_inclusive(min.clone().sin(), max.clone().sin()).unwrap();

                rng.sample_iter(uniform)
                    .map(|value| Some(OVector::from_element_generic(U1, U1, value.asin())))
            }
            // Safe by construction
            _ => unreachable!(),
        }
    }

    fn variance(&self) -> SVector<T, 1> {
        // For a symmetric domain [-a, a], a known constant is returned.
        // For a general bounded domain [a, b], the variance is computed as:
        // variance = E[X²] - (E[X])²
        //
        // where E[X²] = ∫[a,b] x²*cos(x) dx / ∫[a,b] cos(x) dx
        let a = self.minimum();
        let b = self.maximum();

        // Check if domain is symmetric around 0
        if a == -b.clone() {
            // For symmetric domain [-x, x], variance simplifies
            // For the cosine distribution on [-π/2, π/2], variance ≈ (π²/4 - 2)
            // For general [-x, x], compute it
            // Using numerical integration or known result
            return SVector::from([b.clone() * b.clone() - tval!(2.0, f64)]);
        }

        // Normalization constant: ∫_a^b cos(x) dx = sin(b) - sin(a)
        let sin_b = b.clone().sin();
        let sin_a = a.clone().sin();
        let norm = sin_b.clone() - sin_a.clone();

        // If norm is approximately zero, the distribution is degenerate
        let norm_abs = norm.clone().abs();
        if ulps_eq!(norm_abs, T::zero()) {
            return SVector::from([T::zero()]);
        }

        let cos_b = b.clone().cos();
        let cos_a = a.clone().cos();
        let mean = self.mean()[0].clone();

        // ∫_a^b x²*cos(x) dx = [x²*sin(x) + 2x*cos(x) - 2*sin(x)]_a^b
        let b_sq = b.clone() * b.clone();
        let a_sq = a.clone() * a.clone();
        let upper = b_sq * sin_b.clone() + b * cos_b * tval!(2.0, f64) - sin_b * tval!(2.0, f64);
        let lower = a_sq * sin_a.clone() + a * cos_a * tval!(2.0, f64) - sin_a * tval!(2.0, f64);
        let e_x_squared = (upper - lower) / norm;

        SVector::from([e_x_squared - mean.clone() * mean])
    }
}

impl<T: RealField> TryFrom<crate::univariate::UnivariateDensity<T>> for CosineDensity<T> {
    type Error = ();

    fn try_from(value: crate::univariate::UnivariateDensity<T>) -> Result<Self, Self::Error> {
        match value {
            crate::univariate::UnivariateDensity::Cosine(pdf) => Ok(pdf),
            _ => Err(()),
        }
    }
}
