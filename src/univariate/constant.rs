use std::iter::repeat;

use crate::{Density, SamplingMode, domain::Domain};
use nalgebra::{Dim, OVector, RealField, SVector, Scalar, U1, VectorView};
use rand::RngExt;
use serde::{Deserialize, Serialize};

/// A constant (point mass) PDF.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ConstantDensity<T>(Domain<T, U1>)
where
    T: Scalar;

impl<T> ConstantDensity<T>
where
    T: RealField,
{
    /// Create a new [`ConstantDensity`].
    pub fn new(constant: T) -> Self {
        Self(Domain::new_mdomain(OVector::from_element_generic(
            U1,
            U1,
            (Some(constant.clone()), Some(constant)),
        )))
    }

    /// Returns the constant value.
    pub fn constant(&self) -> T {
        match &self.0.inner().unwrap() {
            (Some(constant), Some(_)) => constant.clone(),
            // Safe: ConstantDensity constructor creates MDomain with (c, c) bounds,
            // ensuring both min and max are always Some(c).
            _ => unreachable!("ConstantDensity MDomain always has explicit equal bounds"),
        }
    }
}

impl<T> Density<T, U1> for ConstantDensity<T>
where
    T: RealField,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, U1, RStride, CStride>,
    ) -> Option<T> {
        if !self.0.contains(sample) {
            return None;
        }

        Some(T::one())
    }

    fn domain(&self) -> Domain<T, U1> {
        self.0.clone()
    }

    fn mean(&self) -> SVector<T, 1> {
        SVector::from([self.constant()])
    }

    fn sample(&self, _rng: &mut impl RngExt, _mode: &SamplingMode) -> Option<SVector<T, 1>> {
        match &self.0.inner().unwrap() {
            (Some(constant), Some(_)) => Some(SVector::from([constant.clone()])),
            // Safe by construction
            _ => unreachable!(),
        }
    }

    fn sample_iter(&self, _rng: &mut impl RngExt) -> impl Iterator<Item = Option<SVector<T, 1>>> {
        match &self.0.inner().unwrap() {
            (Some(constant), Some(_)) => repeat(Some(OVector::from([constant.clone()]))),
            // Safe by construction
            _ => unreachable!(),
        }
    }

    fn variance(&self) -> SVector<T, 1> {
        SVector::from([T::zero()])
    }
}

impl<T: RealField> TryFrom<crate::univariate::UnivariateDensity<T>> for ConstantDensity<T> {
    type Error = ();

    fn try_from(value: crate::univariate::UnivariateDensity<T>) -> Result<Self, Self::Error> {
        match value {
            crate::univariate::UnivariateDensity::Constant(pdf) => Ok(pdf),
            _ => Err(()),
        }
    }
}
