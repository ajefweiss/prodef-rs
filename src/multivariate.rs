//! A module that implements multiple univariate PDFs and a multivariate distribution that is a product of independent univariate PDFs.

use crate::{Density, UnivariateDensity, domain::Domain, tval, univariate::match_univariate};
use derive_more::IntoIterator;
use nalgebra::{
    DefaultAllocator, Dim, OVector, RealField, SVector, Scalar, U1, VectorView,
    allocator::Allocator,
};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal, StandardUniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};
use std::{f64, fmt::Debug, iter::repeat_with};

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
/// # use prodef::{ConstantDensity, MultivariateDensity, NormalDensity, UniformDensity, Density};
/// # use rand::{SeedableRng, rngs::StdRng};
/// let marginals = SVector::from([
///     NormalDensity::new(0.0, 1.0, None, None).unwrap().into(),
///     UniformDensity::new(-1.0, 1.0).unwrap().into(),
/// ]);
/// let dist = MultivariateDensity::<f64, U2>::new(marginals);
/// let mut rng = StdRng::seed_from_u64(42);
/// if let Some(sample) = (&dist).sample(&mut rng) {
///     println!("Generated sample: {:?}", sample);
/// }
/// ```
#[derive(Clone, Debug, Deserialize, IntoIterator, Serialize)]
#[serde(bound(serialize = "OVector<UnivariateDensity<T>, D>: Serialize"))]
#[serde(bound(deserialize = "OVector<UnivariateDensity<T>, D>: Deserialize<'de>"))]
pub struct MultivariateDensity<T, D>(#[into_iterator(owned, ref)] OVector<UnivariateDensity<T>, D>)
where
    T: Scalar,
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

    /// Return a reference to the underlying vector of [`UnivariateDensity`]s.
    pub fn marginals(&self) -> &OVector<UnivariateDensity<T>, D> {
        &self.0
    }
}

impl<T, D> Density<T, D> for MultivariateDensity<T, D>
where
    T: RealField + SampleUniform,
    D: Dim,
    DefaultAllocator: Allocator<D>,
    StandardNormal: Distribution<T>,
    StandardUniform: Distribution<T>,
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
                Density::<T, U1>::density::<U1, U1>(pdf, &vec.as_view())
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
                let (lower_bound, upper_bound) = match uvpdf {
                    UnivariateDensity::Constant(pdf) => {
                        (Some(pdf.constant()), Some(pdf.constant()))
                    }
                    UnivariateDensity::Cosine(pdf) => (Some(pdf.minimum()), Some(pdf.maximum())),
                    UnivariateDensity::Lognormal(pdf) => (Some(pdf.minimum()), Some(pdf.maximum())),
                    UnivariateDensity::Loguniform(pdf) => {
                        (Some(pdf.minimum()), Some(pdf.maximum()))
                    }
                    UnivariateDensity::Normal(pdf) => (pdf.minimum(), pdf.maximum()),
                    UnivariateDensity::Uniform(pdf) => (Some(pdf.minimum()), Some(pdf.maximum())),
                };

                (lower_bound, upper_bound)
            }),
        ))
    }

    fn mean(&self) -> OVector<T, D> {
        OVector::from_iterator_generic(
            self.0.shape_generic().0,
            U1,
            self.0
                .iter()
                .map(|uvpdf| match_univariate!(uvpdf, pdf, { pdf.mean() })[0].clone()),
        )
    }

    fn sample<R>(&self, rng: &mut R) -> Option<OVector<T, D>>
    where
        R: RngExt + SeedableRng,
    {
        let mut draw = OVector::<T, D>::zeros_generic(self.0.shape_generic().0, U1);

        for idx in 0..self.0.shape_generic().0.value() {
            draw[idx] = match_univariate!(&self.0[idx], pdf, {
                let result = match Density::<T, U1>::sample(pdf, &mut rng.fork()) {
                    Some(sample) => sample[0].clone(),
                    None => return None,
                };

                result
            });
        }

        Some(draw)
    }

    fn sample_iter<R>(&self, rng: &mut R) -> impl Iterator<Item = Option<OVector<T, D>>>
    where
        R: RngExt + SeedableRng,
    {
        let n_dim = self.0.shape_generic().0;

        repeat_with(move || {
            let mut draw = OVector::<T, D>::zeros_generic(n_dim, U1);

            for (idx, pdf) in self.into_iter().enumerate() {
                let sample = pdf.sample(&mut rng.fork())?;
                draw[idx] = sample[0].clone();
            }

            Some(draw)
        })
    }

    fn variance(&self) -> OVector<T, D> {
        OVector::from_iterator_generic(
            self.0.shape_generic().0,
            U1,
            self.0
                .iter()
                .map(|uvpdf| match_univariate!(uvpdf, pdf, { pdf.variance() })[0].clone()),
        )
    }
}
