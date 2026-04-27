use crate::{Density, RejectionSampler, SamplingMode, domain::Domain, macros::tval};
use nalgebra::{Dim, OVector, RealField, SVector, U1, VectorView};
use rand::RngExt;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

/// A univariate normal PDF.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NormalDensity<T>(T, T, Domain<T, U1>)
where
    T: RealField;

impl<T> NormalDensity<T>
where
    T: RealField,
{
    /// Evaluates the cumulative distribution function at `x`.
    pub fn cdf(&self, x: T) -> T {
        let z = (x - self.0.clone()) / (self.1.clone() * tval!(2, usize).sqrt());

        tval!(0.5, f64) * (T::one() + Self::erf(z))
    }

    /// Evaluates the error function at `x`.
    pub fn erf(z: T) -> T {
        tval!(2, usize) / T::pi().sqrt()
            * (z.clone() - z.clone().powi(3) / tval!(3, usize)
                + z.clone().clone().powi(5) / tval!(10, usize)
                - z.clone().powi(7) / tval!(42, usize)
                + z.clone().powi(9) / tval!(216, usize)
                - z.clone().powi(11) / tval!(1320, usize))
    }

    /// Create a new [`NormalDensity`].
    pub fn new(mean: T, std_dev: T, opt_a: Option<T>, opt_b: Option<T>) -> Option<Self> {
        if std_dev <= T::zero() {
            return None;
        }

        let sdom = (opt_a.clone(), opt_b.clone());

        if opt_a.unwrap_or(T::neg(T::one())) >= opt_b.unwrap_or(T::one()) {
            return None;
        }

        let domain = Domain::new_mdomain(OVector::from_element_generic(U1, U1, sdom));

        Some(Self(mean, std_dev, domain))
    }

    /// Returns the maximum value of the domain.
    pub fn maximum(&self) -> Option<T> {
        match &self.2.inner().unwrap() {
            (_, Some(max)) => Some(max.clone()),
            _ => None,
        }
    }

    /// Returns the minimum value of the domain.
    pub fn minimum(&self) -> Option<T> {
        match &self.2.inner().unwrap() {
            (Some(min), _) => Some(min.clone()),
            _ => None,
        }
    }
}

impl<T> Density<T, U1> for &NormalDensity<T>
where
    T: RealField,
    StandardNormal: Distribution<T>,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, U1, RStride, CStride>,
    ) -> Option<T> {
        if !self.2.contains(sample) {
            return None;
        }

        Some(
            T::one() / (self.1.clone() * tval!(2.0 * std::f64::consts::PI, f64).sqrt())
                * (-((sample[0].clone() - self.0.clone()) / self.1.clone()).powi(2)
                    / tval!(2, usize))
                .exp(),
        )
    }

    fn domain(&self) -> Domain<T, U1> {
        self.2.clone()
    }

    fn mean(&self) -> SVector<T, 1> {
        // For an unbounded normal distribution, returns the parameter μ.
        // For a bounded normal distribution, returns an approximation accounting for truncation.
        // If μ is within the domain, returns μ. Otherwise, returns the boundary point closer to μ.
        let mu = self.0.clone();
        let a = self.minimum();
        let b = self.maximum();

        // If domain is unbounded or μ is within bounds, return μ
        if let (Some(min), Some(max)) = (a.clone(), b.clone()) {
            if min <= mu.clone() && mu.clone() <= max {
                return SVector::from([mu]);
            }
            // Both bounds exist, mean is outside. Return closer boundary.
            if mu.clone() < min {
                return SVector::from([min]);
            } else {
                return SVector::from([max]);
            }
        }

        // Only one bound exists
        if let Some(min) = a.clone()
            && mu.clone() < min
        {
            return SVector::from([min]);
        }
        if let Some(max) = b.clone()
            && mu.clone() > max
        {
            return SVector::from([max]);
        }

        SVector::from([mu])
    }

    fn sample(&self, rng: &mut impl RngExt, mode: &SamplingMode) -> Option<SVector<T, 1>> {
        self.rejection_sample(rng, mode)
    }

    fn sample_iter(&self, rng: &mut impl RngExt) -> impl Iterator<Item = Option<SVector<T, 1>>> {
        let normal = StandardNormal;

        rng.sample_iter(normal).map(move |z| {
            let candidate = self.1.clone() * z + self.0.clone();

            if self
                .2
                .contains::<U1, U1>(&SVector::from([candidate.clone()]).as_view())
            {
                Some(OVector::from([candidate]))
            } else {
                None
            }
        })
    }

    fn variance(&self) -> SVector<T, 1> {
        SVector::from([self.1.clone().powi(2)])
    }
}

impl<T> RejectionSampler<T, U1> for &NormalDensity<T>
where
    T: RealField,
    StandardNormal: Distribution<T>,
{
    fn generate_candidate(&self, rng: &mut impl RngExt) -> SVector<T, 1> {
        let z = rng.sample(StandardNormal);

        OVector::from([self.1.clone() * z + self.0.clone()])
    }
}
