use crate::{Density, domain::Domain, tval};
use nalgebra::{Dim, OVector, RealField, SVector, Scalar, U1, VectorView};
use rand::{RngExt, SeedableRng};
use rand_distr::{Uniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};

/// A uniform PDF.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct UniformDensity<T>(Domain<T, U1>)
where
    T: Scalar;

impl<T> UniformDensity<T>
where
    T: RealField,
{
    /// Create a new [`UniformDensity`].
    pub fn new(lower_bound: T, upper_bound: T) -> Option<Self> {
        if lower_bound >= upper_bound {
            None
        } else {
            Some(Self(Domain::new_mdomain(OVector::from_element_generic(
                U1,
                U1,
                (Some(lower_bound), Some(upper_bound)),
            ))))
        }
    }

    /// Returns the maximum value of the domain.
    pub fn maximum(&self) -> T {
        match &self.0.inner().unwrap() {
            (_, Some(max)) => max.clone(),
            // Safe: UniformDensity constructor creates MDomain with explicit bounds (a, b)
            // where a < b, so upper bound is always Some(max).
            _ => unreachable!("MDomain always has explicit bounds in UniformDensity"),
        }
    }

    /// Returns the minimum value of the domain.
    pub fn minimum(&self) -> T {
        match &self.0.inner().unwrap() {
            (Some(min), _) => min.clone(),
            // Safe: UniformDensity constructor creates MDomain with explicit bounds (a, b)
            // where a < b, so lower bound is always Some(min).
            _ => unreachable!("MDomain always has explicit bounds in UniformDensity"),
        }
    }
}

impl<T> Density<T, U1> for UniformDensity<T>
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

        Some(
            T::one()
                / (self.0.maximum_values()[0].clone().unwrap()
                    - self.0.minimum_values()[0].clone().unwrap()),
        )
    }

    fn domain(&self) -> Domain<T, U1> {
        self.0.clone()
    }

    fn mean(&self) -> SVector<T, 1> {
        SVector::from([(self.minimum() + self.maximum()) / tval!(2, usize)])
    }

    fn sample<R>(&self, rng: &mut R) -> Option<SVector<T, 1>>
    where
        R: RngExt + SeedableRng,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<U1>,
    {
        let uniform = Uniform::new_inclusive(
            self.0.minimum_values()[0].clone().unwrap(),
            self.0.maximum_values()[0].clone().unwrap(),
        )
        .unwrap();

        Some(SVector::from([rng.sample(uniform)]))
    }

    fn sample_iter<R>(&self, rng: &mut R) -> impl Iterator<Item = Option<SVector<T, 1>>>
    where
        R: RngExt + SeedableRng,
        nalgebra::DefaultAllocator: nalgebra::allocator::Allocator<U1>,
    {
        let uniform = Uniform::new_inclusive(
            self.0.minimum_values()[0].clone().unwrap(),
            self.0.maximum_values()[0].clone().unwrap(),
        )
        .unwrap();

        rng.sample_iter(uniform)
            .map(|value| Some(OVector::from([value])))
    }

    fn variance(&self) -> SVector<T, 1> {
        let range = self.maximum() - self.minimum();

        SVector::from([range.powi(2) / tval!(12, usize)])
    }
}

impl<T: RealField> TryFrom<crate::univariate::UnivariateDensity<T>> for UniformDensity<T> {
    type Error = ();

    fn try_from(value: crate::univariate::UnivariateDensity<T>) -> Result<Self, Self::Error> {
        match value {
            crate::univariate::UnivariateDensity::Uniform(pdf) => Ok(pdf),
            _ => Err(()),
        }
    }
}
