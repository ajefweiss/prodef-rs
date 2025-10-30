//! A module that implements multiple univariate PDFs and a multivariate distribution that is a product of independent univariate PDFs.

mod constant;
mod cosine;
mod loguniform;
mod normal;
mod uniform;

pub use constant::*;
pub use cosine::*;
pub use loguniform::*;
pub use normal::*;
pub use uniform::*;

use crate::{Density, SamplingMode, domain::Domain, macros::tval};
use derive_more::IntoIterator;
use nalgebra::{
    DefaultAllocator, Dim, OVector, RealField, SVector, U1, VectorView, allocator::Allocator,
};
use rand::RngExt;
use rand_distr::{Distribution, StandardNormal, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::{f64, fmt::Debug, iter::repeat_with};

/// A macro to match on a UnivariateDensity and apply a closure to its inner variant.
macro_rules! match_univariate {
    ($uvpdf:expr, $pat:pat, $body:expr) => {
        match $uvpdf {
            UnivariateDensity::Constant($pat) => $body,
            UnivariateDensity::Cosine($pat) => $body,
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
    T: RealField,
{
    /// A constant PDF.
    Constant(ConstantDensity<T>),
    /// A cosine PDF.
    Cosine(CosineDensity<T>),
    /// A log-uniform PDF.
    Loguniform(LogUniformDensity<T>),
    /// A normal PDF.
    Normal(NormalDensity<T>),
    /// A uniform PDF.
    Uniform(UniformDensity<T>),
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

impl<T> UnivariateDensity<T>
where
    T: RealField,
{
    /// Returns a reference to the inner [`ConstantDensity`] if this is a `UnivariateDensity::Constant`, `None` otherwise.
    pub fn as_constant(&self) -> Option<&ConstantDensity<T>> {
        match self {
            UnivariateDensity::Constant(pdf) => Some(pdf),
            _ => None,
        }
    }

    /// Returns a reference to the inner [`CosineDensity`] if this is a `UnivariateDensity::Cosine`, `None` otherwise.
    pub fn as_cosine(&self) -> Option<&CosineDensity<T>> {
        match self {
            UnivariateDensity::Cosine(pdf) => Some(pdf),
            _ => None,
        }
    }

    /// Returns a reference to the inner [`LogUniformDensity`] if this is a `UnivariateDensity::Loguniform`, `None` otherwise.
    pub fn as_loguniform(&self) -> Option<&LogUniformDensity<T>> {
        match self {
            UnivariateDensity::Loguniform(pdf) => Some(pdf),
            _ => None,
        }
    }

    /// Returns a reference to the inner [`NormalDensity`] if this is a `UnivariateDensity::Normal`, `None` otherwise.
    pub fn as_normal(&self) -> Option<&NormalDensity<T>> {
        match self {
            UnivariateDensity::Normal(pdf) => Some(pdf),
            _ => None,
        }
    }

    /// Returns a reference to the inner [`UniformDensity`] if this is a `UnivariateDensity::Uniform`, `None` otherwise.
    pub fn as_uniform(&self) -> Option<&UniformDensity<T>> {
        match self {
            UnivariateDensity::Uniform(pdf) => Some(pdf),
            _ => None,
        }
    }
}

impl<T> Density<T, U1> for &UnivariateDensity<T>
where
    T: RealField + SampleUniform,
    StandardNormal: Distribution<T>,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, U1, RStride, CStride>,
    ) -> Option<T> {
        match_univariate!(self, pdf, {
            Density::<T, U1>::density::<RStride, CStride>(&pdf, sample)
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

    fn sample(&self, rng: &mut impl RngExt, mode: &SamplingMode) -> Option<OVector<T, U1>> {
        let sample = match_univariate!(self, pdf, {
            match Density::<T, U1>::sample(&pdf, rng, mode) {
                Some(draw) => draw[0].clone(),
                None => return None,
            }
        });

        Some(OVector::from([sample]))
    }

    fn sample_iter(&self, rng: &mut impl RngExt) -> impl Iterator<Item = Option<OVector<T, U1>>> {
        // Likely not very efficient, but the only way to have a unique opaque return type.
        repeat_with(move || {
            match_univariate!(self, pdf, {
                Density::<T, U1>::sample(&pdf, rng, &SamplingMode::SingleAttempt)
                    .map(|value| OVector::from([value[0].clone()]))
            })
        })
    }
}

/// A `D`-dimensional distribution where each dimension is
/// **independent** with potentially different univariate distributions. This is a **product distribution**:
/// - Each marginal follows one of the available univariate distributions (Normal, Uniform, Cosine, etc.)
/// - The joint density is the product of marginals: f(x₁, ..., xₐ) = f₁(x₁) × ... × fₐ(xₐ)
///
/// # Construction & Examples
///
/// Create a mixed 3D distribution (Normal × Uniform × Constant):
/// ```
/// # use nalgebra::{Const, SVector};
/// # use prodef::{ConstantDensity, MultivariateDensity, NormalDensity, UniformDensity, Density};
/// let marginals = SVector::from([
///     NormalDensity::new(0.0, 1.0, None, None).unwrap().into(),
///     UniformDensity::new(-1.0, 1.0).unwrap().into(),
///     ConstantDensity::new(2.0).into(),
/// ]);
/// let _dist = MultivariateDensity::<f64, Const<3>>::new(marginals);
/// ```
///
/// Create a 5D distribution with mixed univariates:
/// ```
/// # use nalgebra::{Const, SVector};
/// # use prodef::{ConstantDensity, CosineDensity, LogUniformDensity, MultivariateDensity, NormalDensity, UniformDensity};
/// let mvpdf = MultivariateDensity::<f64, Const<5>>::new(SVector::from([
///    ConstantDensity::new(1.0).into(),
///    CosineDensity::new(0.1, 0.2).unwrap().into(),
///    LogUniformDensity::new(0.1, 0.5).unwrap().into(),
///    NormalDensity::new(0.1, 0.25, Some(-0.5), Some(1.5)).unwrap().into(),
///    UniformDensity::new(1.0, 2.0).unwrap().into(),
/// ]));
/// ```
///
/// Evaluate density at a point:
/// ```
/// # use nalgebra::{U1, U2, SVector};
/// # use prodef::{ConstantDensity, MultivariateDensity, NormalDensity, UniformDensity, Density};
/// let marginals = SVector::from([
///     NormalDensity::new(0.0, 1.0, None, None).unwrap().into(),
///     UniformDensity::new(-1.0, 1.0).unwrap().into(),
/// ]);
/// let dist = MultivariateDensity::<f64, U2>::new(marginals);
/// let sample = SVector::from([0.0, 0.5]);
/// // Use the Density trait to evaluate - see crate::Density for usage patterns
/// if let Some(dens) = (&dist).density::<U1, U2>(&sample.as_view()) {
///     println!("Joint density: {}", dens);
/// }
/// ```
///
/// Sample from the distribution:
/// ```
/// # use nalgebra::{U2, SVector};
/// # use prodef::{ConstantDensity, MultivariateDensity, NormalDensity, UniformDensity, Density, SamplingMode};
/// # use rand::{SeedableRng, rngs::StdRng};
/// let marginals = SVector::from([
///     NormalDensity::new(0.0, 1.0, None, None).unwrap().into(),
///     UniformDensity::new(-1.0, 1.0).unwrap().into(),
/// ]);
/// let dist = MultivariateDensity::<f64, U2>::new(marginals);
/// let mut rng = StdRng::seed_from_u64(42);
/// if let Some(sample) = (&dist).sample(&mut rng, &SamplingMode::default()) {
///     println!("Generated sample: {:?}", sample);
/// }
/// ```
#[derive(Clone, Debug, Deserialize, IntoIterator, Serialize)]
#[serde(bound(serialize = "OVector<UnivariateDensity<T>, D>: Serialize"))]
#[serde(bound(deserialize = "OVector<UnivariateDensity<T>, D>: Deserialize<'de>"))]
pub struct MultivariateDensity<T, D>(#[into_iterator(owned, ref)] OVector<UnivariateDensity<T>, D>)
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D>;

impl<T, D> MultivariateDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Create a new [`MultivariateDensity`] from a vector of [`UnivariateDensity`]s.
    pub fn new(domains: OVector<UnivariateDensity<T>, D>) -> Self {
        Self(domains)
    }
}

impl<T, D> Density<T, D> for &MultivariateDensity<T, D>
where
    T: RealField + SampleUniform,
    D: Dim,
    StandardNormal: Distribution<T>,
    DefaultAllocator: Allocator<D>,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, D, RStride, CStride>,
    ) -> Option<T> {
        if !self.domain().contains(sample) {
            return None;
        }

        let mut rlh = T::one();

        self.0.iter().zip(sample.iter()).for_each(|(uvpdf, value)| {
            let vec = SVector::from([value.clone()]);

            rlh *= match_univariate!(uvpdf, pdf, {
                Density::<T, U1>::density::<U1, U1>(&pdf, &vec.as_view())
            })
            .unwrap_or(tval!(f64::NAN, f64));
        });

        Some(rlh)
    }

    fn domain(&self) -> Domain<T, D> {
        Domain::new_mdomain(OVector::from_iterator_generic(
            self.0.shape_generic().0,
            U1,
            self.0.iter().map(|uvpdf| {
                let (a, b) = match uvpdf {
                    UnivariateDensity::Constant(pdf) => {
                        (Some(pdf.constant()), Some(pdf.constant()))
                    }
                    UnivariateDensity::Cosine(pdf) => (Some(pdf.minimum()), Some(pdf.maximum())),
                    UnivariateDensity::Loguniform(pdf) => {
                        (Some(pdf.minimum()), Some(pdf.maximum()))
                    }
                    UnivariateDensity::Normal(pdf) => (pdf.minimum(), pdf.maximum()),
                    UnivariateDensity::Uniform(pdf) => (Some(pdf.minimum()), Some(pdf.maximum())),
                };

                (a, b)
            }),
        ))
    }

    fn sample(&self, rng: &mut impl RngExt, mode: &SamplingMode) -> Option<OVector<T, D>> {
        let mut draw = OVector::<T, D>::zeros_generic(self.0.shape_generic().0, U1);

        for i in 0..self.0.shape_generic().0.value() {
            draw[i] = match_univariate!(&self.0[i], pdf, {
                match Density::<T, U1>::sample(&pdf, rng, mode) {
                    Some(sample) => sample[0].clone(),
                    None => return None,
                }
            });
        }

        Some(draw)
    }

    fn sample_iter(&self, rng: &mut impl RngExt) -> impl Iterator<Item = Option<OVector<T, D>>> {
        let n_dim = self.0.shape_generic().0;

        repeat_with(move || {
            let draw_opts = OVector::<Option<SVector<T, 1>>, D>::from_iterator_generic(
                n_dim,
                U1,
                self.into_iter()
                    .map(|pdf| pdf.sample(rng, &SamplingMode::SingleAttempt)),
            );

            if draw_opts.iter().any(|draw| draw.is_none()) {
                return None;
            }

            // All samples are guaranteed to be Some due to check above
            let draw = OVector::<T, D>::from_iterator_generic(
                n_dim,
                U1,
                draw_opts.iter().map(|opt_draw| {
                    // Safe: we verified no None values exist above
                    opt_draw.clone().unwrap()[0].clone()
                }),
            );

            Some(draw)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;
    use nalgebra::{OVector, SVector, U2, U5};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_multivariate_density_independent_evaluation() {
        let const_a = ConstantDensity::new(1.0);
        let const_b = ConstantDensity::new(2.0);

        assert!(ulps_eq!(
            (&const_a)
                .density::<U1, U1>(&OVector::from([1.0]).as_view())
                .unwrap(),
            (&const_b)
                .density::<U1, U1>(&OVector::from([2.0]).as_view())
                .unwrap()
        ));

        let cosine = CosineDensity::new(-0.1, 0.35).unwrap();

        assert!(ulps_eq!(
            (&cosine)
                .density::<U1, U1>(&OVector::from([0.0]).as_view())
                .unwrap(),
            1.0
        ));

        assert!(ulps_eq!(
            (&cosine)
                .density::<U1, U1>(&OVector::from([0.3]).as_view())
                .unwrap(),
            0.955336489125606
        ));

        assert!(ulps_eq!(
            (&cosine)
                .density::<U1, U1>(&OVector::from([0.3]).as_view())
                .unwrap(),
            0.955336489125606
        ));

        let loguniform = LogUniformDensity::new(0.5, 2.5).unwrap();

        assert!(ulps_eq!(
            (&loguniform)
                .density::<U1, U1>(&OVector::from([1.0]).as_view())
                .unwrap(),
            2.0 * (&loguniform)
                .density::<U1, U1>(&OVector::from([2.0]).as_view())
                .unwrap()
        ));

        let normal = NormalDensity::new(1.0, 1.0, None, None).unwrap();

        assert!(ulps_eq!(
            (&normal)
                .density::<U1, U1>(&OVector::from([1.0]).as_view())
                .unwrap(),
            0.3989422804014327
        ));

        assert!(ulps_eq!(
            (&normal)
                .density::<U1, U1>(&OVector::from([2.0]).as_view())
                .unwrap(),
            0.24197072451914337
        ));

        let uniform = UniformDensity::new(0.0, 1.0).unwrap();

        assert!(ulps_eq!(
            (&uniform)
                .density::<U1, U1>(&OVector::from([0.5]).as_view())
                .unwrap(),
            1.0
        ));

        let uvpdf = &MultivariateDensity::new(SVector::from([
            ConstantDensity::new(1.0).into(),
            CosineDensity::new(0.1, 0.2).unwrap().into(),
            LogUniformDensity::new(0.1, 0.5).unwrap().into(),
            NormalDensity::new(0.1, 0.25, Some(-0.5), Some(1.5))
                .unwrap()
                .into(),
            UniformDensity::new(1.0, 2.0).unwrap().into(),
        ]));

        assert!(ulps_eq!(
            uvpdf
                .density::<U1, U5>(&SVector::from([1.0f64, 0.15, 0.15, 0.2, 1.5]).as_view())
                .unwrap(),
            6.033325,
            epsilon = 1e-5,
            max_ulps = 5
        ));

        assert!(
            uvpdf
                .density::<U1, U5>(&SVector::from([1.0, 0.05, 0.2, 0.15, 1.5]).as_view())
                .is_none()
        );

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);

        assert!(ulps_eq!(
            uvpdf
                .sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 100 })
                .unwrap(),
            OVector::from([1.0, 0.1810371, 0.33281568, -0.37896788, 1.7462168,]),
            epsilon = 1e-5,
            max_ulps = 5
        ));

        assert!(
            uvpdf.domain().contains::<U1, U5>(
                &uvpdf
                    .sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 100 })
                    .unwrap()
                    .as_view()
            )
        );
    }

    #[test]
    fn test_multivariate_sample_iter_independent_marginals() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mvpdf = &MultivariateDensity::new(SVector::from([
            NormalDensity::new(0.0, 1.0, None, None).unwrap().into(),
            UniformDensity::new(0.0, 1.0).unwrap().into(),
        ]));

        let samples: Vec<_> = mvpdf.sample_iter(&mut rng).take(100).flatten().collect();

        assert_eq!(samples.len(), 100);

        // Each marginal should have appropriate statistics
        let mean_0: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        let mean_1: f64 = samples.iter().map(|s| s[1]).sum::<f64>() / samples.len() as f64;

        // Normal(0,1) mean should be close to 0
        assert!(mean_0.abs() < 0.3);

        // Uniform(0,1) mean should be close to 0.5
        assert!((mean_1 - 0.5).abs() < 0.15);

        // Uniform samples should be in [0, 1]
        for sample in &samples {
            assert!(sample[1] >= 0.0 && sample[1] <= 1.0);
        }
    }

    #[test]
    fn test_multivariate_sample_iter_5d() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        let mvpdf = &MultivariateDensity::new(SVector::from([
            ConstantDensity::new(1.0).into(),
            CosineDensity::new(0.1, 0.2).unwrap().into(),
            LogUniformDensity::new(0.1, 0.5).unwrap().into(),
            NormalDensity::new(0.1, 0.25, Some(-0.5), Some(1.5))
                .unwrap()
                .into(),
            UniformDensity::new(1.0, 2.0).unwrap().into(),
        ]));

        let samples: Vec<_> = mvpdf.sample_iter(&mut rng).take(50).flatten().collect();

        assert!(!samples.is_empty());

        // Verify all samples are in valid domain
        for sample in &samples {
            assert!(mvpdf.domain().contains::<U1, U5>(&sample.as_view()));
        }
    }

    #[test]
    fn test_multivariate_sample_iter_domain_enforcement() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mvpdf = &MultivariateDensity::new(SVector::from([
            NormalDensity::new(0.0, 1.0, Some(-1.0), Some(1.0))
                .unwrap()
                .into(),
            UniformDensity::new(0.0, 1.0).unwrap().into(),
        ]));

        let results: Vec<_> = mvpdf.sample_iter(&mut rng).take(200).collect();

        // Some should be None due to normal distribution truncation
        let none_count = results.iter().filter(|r| r.is_none()).count();
        assert!(
            none_count > 0,
            "Expected some rejections due to domain constraints"
        );

        // All Some samples should satisfy domain
        for result in results.iter().flatten() {
            assert!(mvpdf.domain().contains::<U1, U2>(&result.as_view()));
        }
    }
}
