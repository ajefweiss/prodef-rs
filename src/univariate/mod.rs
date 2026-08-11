//! A module that implements multiple univariate PDFs and a multivariate distribution that is a product of independent univariate PDFs.

mod constant;
mod cosine;
mod lognormal;
mod loguniform;
mod normal;
mod uniform;

pub use constant::*;
pub use cosine::*;
pub use lognormal::*;
pub use loguniform::*;
pub use normal::*;
pub use uniform::*;

use crate::{Density, MultivariateDensity, domain::Domain};
use nalgebra::{
    DefaultAllocator, Dim, OVector, RealField, SVector, Scalar, U1, VectorView,
    allocator::Allocator,
};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal, StandardUniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, iter::repeat_with};

/// A macro to match on a UnivariateDensity and apply a closure to its inner variant.
macro_rules! match_univariate {
    ($uvpdf:expr, $pat:pat, $body:expr) => {
        match $uvpdf {
            UnivariateDensity::Constant($pat) => $body,
            UnivariateDensity::Cosine($pat) => $body,
            UnivariateDensity::Lognormal($pat) => $body,
            UnivariateDensity::Loguniform($pat) => $body,
            UnivariateDensity::Normal($pat) => $body,
            UnivariateDensity::Uniform($pat) => $body,
        }
    };
}

/// An algebraic data type (ADT) that contains all univariate PDFs.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", content = "content")]
pub enum UnivariateDensity<T>
where
    T: Scalar,
{
    /// A constant PDF.
    Constant(ConstantDensity<T>),
    /// A cosine PDF.
    Cosine(CosineDensity<T>),
    /// A lognormal PDF.
    Lognormal(LognormalDensity<T>),
    /// A log-uniform PDF.
    Loguniform(LogUniformDensity<T>),
    /// A normal PDF.
    Normal(NormalDensity<T>),
    /// A uniform PDF.
    Uniform(UniformDensity<T>),
}

impl<T> Density<T, U1> for UnivariateDensity<T>
where
    T: RealField + SampleUniform,
    StandardNormal: Distribution<T>,
    StandardUniform: Distribution<T>,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, U1, RStride, CStride>,
    ) -> Option<T> {
        match_univariate!(self, pdf, {
            Density::<T, U1>::density::<RStride, CStride>(pdf, sample)
        })
    }

    fn domain(&self) -> Domain<T, U1> {
        let (a, b) = match_univariate!(self, pdf, {
            (
                pdf.domain().minimum_values()[0].clone(),
                pdf.domain().maximum_values()[0].clone(),
            )
        });

        Domain::new_mdomain(OVector::from_element_generic(U1, U1, (a, b)))
    }

    fn mean(&self) -> SVector<T, 1> {
        match_univariate!(self, pdf, { pdf.mean() })
    }

    fn sample<R>(&self, rng: &mut R) -> Option<SVector<T, 1>>
    where
        R: RngExt + SeedableRng,
        DefaultAllocator: Allocator<U1>,
    {
        let sample = match_univariate!(self, pdf, {
            match Density::<T, U1>::sample(pdf, rng) {
                Some(draw) => draw[0].clone(),
                None => return None,
            }
        });

        Some(OVector::from([sample]))
    }

    fn sample_iter<R>(&self, rng: &mut R) -> impl Iterator<Item = Option<SVector<T, 1>>>
    where
        R: RngExt + SeedableRng,
        DefaultAllocator: Allocator<U1>,
    {
        // Likely not very efficient, but the only way to have a unique opaque return type.
        repeat_with(move || {
            match_univariate!(self, pdf, {
                Density::<T, U1>::sample(pdf, rng).map(|value| OVector::from([value[0].clone()]))
            })
        })
    }

    fn variance(&self) -> SVector<T, 1> {
        match_univariate!(self, pdf, { pdf.variance() })
    }
}

impl<T> From<ConstantDensity<T>> for UnivariateDensity<T>
where
    T: RealField,
{
    fn from(value: ConstantDensity<T>) -> Self {
        Self::Constant(value)
    }
}

impl<T> From<CosineDensity<T>> for UnivariateDensity<T>
where
    T: RealField,
{
    fn from(value: CosineDensity<T>) -> Self {
        Self::Cosine(value)
    }
}

impl<T> From<LognormalDensity<T>> for UnivariateDensity<T>
where
    T: RealField,
{
    fn from(value: LognormalDensity<T>) -> Self {
        Self::Lognormal(value)
    }
}

impl<T> From<LogUniformDensity<T>> for UnivariateDensity<T>
where
    T: RealField,
{
    fn from(value: LogUniformDensity<T>) -> Self {
        Self::Loguniform(value)
    }
}

impl<T> From<NormalDensity<T>> for UnivariateDensity<T>
where
    T: RealField,
{
    fn from(value: NormalDensity<T>) -> Self {
        Self::Normal(value)
    }
}

impl<T> From<UniformDensity<T>> for UnivariateDensity<T>
where
    T: RealField,
{
    fn from(value: UniformDensity<T>) -> Self {
        Self::Uniform(value)
    }
}

impl<T> From<MultivariateDensity<T, U1>> for UnivariateDensity<T>
where
    T: RealField,
{
    /// Convert a 1D [`MultivariateDensity`] to a [`UnivariateDensity`] by extracting its single marginal.
    fn from(mv: MultivariateDensity<T, U1>) -> Self {
        // Extract the single univariate from the 1D multivariate
        mv.marginals()[0].clone()
    }
}

pub(crate) use match_univariate;
