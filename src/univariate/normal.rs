use crate::{Density, domain::Domain, sampling::RejectionSampling, tval};
use nalgebra::{Dim, OVector, RealField, SVector, Scalar, U1, VectorView};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal, StandardUniform};
use serde::{Deserialize, Serialize};

/// A univariate normal PDF.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NormalDensity<T>(T, T, Domain<T, U1>)
where
    T: Scalar;

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
                + z.clone().powi(5) / tval!(10, usize)
                - z.clone().powi(7) / tval!(42, usize)
                + z.clone().powi(9) / tval!(216, usize)
                - z.powi(11) / tval!(1320, usize))
    }

    /// Create a new [`NormalDensity`].
    pub fn new(
        mean: T,
        std_dev: T,
        lower_bound: Option<T>,
        upper_bound: Option<T>,
    ) -> Option<Self> {
        if std_dev <= T::zero() {
            return None;
        }

        if let (Some(lower), Some(upper)) = (&lower_bound, &upper_bound) {
            if lower >= upper {
                return None;
            }
        }

        let domain = Domain::new_mdomain(OVector::from_element_generic(
            U1,
            U1,
            (lower_bound, upper_bound),
        ));

        Some(Self(mean, std_dev, domain))
    }

    /// Returns the maximum value of the domain.
    pub fn maximum(&self) -> Option<T> {
        self.2.inner().and_then(|(_, max)| max.clone())
    }

    /// Returns the minimum value of the domain.
    pub fn minimum(&self) -> Option<T> {
        self.2.inner().and_then(|(min, _)| min.clone())
    }
}

impl<T> Density<T, U1> for NormalDensity<T>
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
        // For an unbounded normal distribution, this is the parameter μ.
        // For a bounded or truncated normal, we return a simple domain-aware
        // approximation: if the nominal mean falls outside the valid interval,
        // return the closer boundary; otherwise keep μ.
        let mu = self.0.clone();
        let lower_bound = self.minimum();
        let upper_bound = self.maximum();

        // If the domain is unbounded or the nominal mean is inside it, keep μ.
        if let (Some(min), Some(max)) = (&lower_bound, &upper_bound) {
            if min <= &mu && &mu <= max {
                return SVector::from([mu]);
            }
            // Both bounds exist and the nominal mean lies outside. Return the
            // boundary that is closer to μ.
            if &mu < min {
                return SVector::from([min.clone()]);
            } else {
                return SVector::from([max.clone()]);
            }
        }

        // Only one bound exists.
        if let Some(min) = &lower_bound
            && &mu < min
        {
            return SVector::from([min.clone()]);
        }
        if let Some(max) = &upper_bound
            && mu > *max
        {
            return SVector::from([max.clone()]);
        }

        SVector::from([mu])
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
        let mu = self.0.clone();
        let sigma = self.1.clone();
        let domain = self.2.clone();

        std::iter::repeat_with(move || {
            let z = rng.sample(StandardNormal);
            let candidate = sigma.clone() * z + mu.clone();

            if domain.contains::<U1, U1>(&SVector::from([candidate.clone()]).as_view()) {
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

impl<T> RejectionSampling<T, U1> for NormalDensity<T>
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

        let candidate = OVector::from([self.1.clone() * z + self.0.clone()]);

        (candidate, None)
    }

    fn scale_factor(&self) -> T {
        T::one()
    }
}

impl<T: RealField> TryFrom<crate::univariate::UnivariateDensity<T>> for NormalDensity<T> {
    type Error = ();

    fn try_from(value: crate::univariate::UnivariateDensity<T>) -> Result<Self, Self::Error> {
        match value {
            crate::univariate::UnivariateDensity::Normal(pdf) => Ok(pdf),
            _ => Err(()),
        }
    }
}
