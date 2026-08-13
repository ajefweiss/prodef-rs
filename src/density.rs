use crate::Domain;
use nalgebra::{DefaultAllocator, Dim, OVector, RealField, VectorView, allocator::Allocator};
use rand::{RngExt, SeedableRng};

/// A trait that is shared by all probability density functions.
///
/// # Overview
///
/// The `Density` trait is the foundational abstraction for all probability distributions
/// in this crate. It provides a unified interface for:
/// - Evaluating PDF values at sample points
/// - Querying the valid domain of the distribution
/// - Generating random samples from the distribution
///
/// # Generics
///
/// - `T`: The scalar type for numerical values (typically `f64` or `f32`).
///   Must implement [`RealField`] for arithmetic operations.
/// - `D`: The dimension of the probability space, using nalgebra's type-level
///   dimension system. Can be compile-time (e.g., `U1`, `U2`) or runtime (`Dyn`).
///
/// # Domain, Normalization, and Sampling
///
/// **Domain**: Defines the valid domain of the PDF, primarily used for truncation.
/// Implementations should return `None` for samples outside their mathematical domain.
///
/// **Normalization**: The returned density value is not necessarily normalized to integrate to 1.
///
/// **Sampling**: The `sample()` and `sample_iter()` methods draw values from the distribution
/// using the caller-provided RNG. When a distribution defines a finite or truncated domain,
/// implementations typically reject out-of-domain candidates and return `None` for unsuccessful
/// attempts. The iterator form yields `None` for failed draws, while successful iterations produce
/// samples within the distribution's valid domain.
///
/// # Stride Generics
///
/// The stride generics (`RStride`, `CStride`) in `density()` enable working with arbitrary
/// memory layouts of input vectors (column-major, row-major, non-contiguous slices, etc.).
/// This can be annoying to specify, but it allows for maximum flexibility.
///
/// # Examples
///
/// Evaluating at a sample:
/// ```
/// # use prodef::Density;
/// # use nalgebra::{SVector, U1};
/// # let normal = prodef::NormalDensity::new(0.0, 1.0, None, None).unwrap();
/// let sample = SVector::from([0.5]);
///
/// if let Some(dens) = (&normal).density::<U1, U1>(&sample.as_view()) {
///     println!("Density at 0.5: {}", dens);
/// } else {
///     println!("Sample outside domain");
/// }
/// ```
///
/// Sampling from a distribution:
/// ```
/// # use prodef::{Density};
/// # use nalgebra::{SVector, U1};
/// # use rand::{SeedableRng, rngs::StdRng};
/// # let normal = prodef::NormalDensity::new(0.0, 1.0, Some(-3.0), Some(3.0)).unwrap();
/// let mut rng = StdRng::seed_from_u64(42);
///
/// if let Some(sample) = (&normal).sample(&mut rng) {
///     println!("Generated sample: {}", sample[0]);
/// }
/// ```
///
/// Working with multivariate distributions:
/// ```
/// # use prodef::MultivariateDensity;
/// # use nalgebra::{OVector, U2};
/// # use prodef::NormalDensity;
/// # use prodef::UniformDensity;
/// let normal_x = NormalDensity::new(0.0, 1.0, None, None).unwrap();
/// let uniform_y = UniformDensity::new(0.0, 1.0).unwrap();
///
/// let multivariate = MultivariateDensity::new(OVector::from([
///     normal_x.into(),
///     uniform_y.into(),
/// ]));
/// ```
pub trait Density<T, D>: Clone
where
    T: RealField,
    D: Dim,
{
    /// Calculates, or estimates, the density for a given sample.
    /// Returns [`None`] if the sample is outside of the function domain.
    ///
    /// Note that the returned value is not necessarily normalized.
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, D, RStride, CStride>,
    ) -> Option<T>;

    /// Returns the dimension of the distribution.
    fn dim(&self) -> D
    where
        DefaultAllocator: Allocator<D>,
    {
        self.domain().shape_generic()
    }

    /// Returns the underlying function [`Domain`].
    fn domain(&self) -> Domain<T, D>
    where
        DefaultAllocator: Allocator<D>;

    /// Calculates, or estimates, the log density for a given sample.
    /// Returns [`None`] if the sample is outside of the function domain.
    ///
    /// Note that the returned value is not necessarily normalized.
    fn log_density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, D, RStride, CStride>,
    ) -> Option<T> {
        self.density(sample).map(|d| d.ln())
    }

    /// Returns the distribution's center as implemented by the concrete type.
    ///
    /// For bounded or truncated distributions, this value may be a domain-aware
    /// approximation rather than the exact theoretical mean.
    fn mean(&self) -> OVector<T, D>
    where
        DefaultAllocator: Allocator<D>;

    /// Returns the dimensionality of the distribution.
    fn ndim(&self) -> usize
    where
        DefaultAllocator: Allocator<D>,
    {
        self.domain().shape_generic().value()
    }

    /// Draw a random sample from the probability density distribution using the provided sampler.
    ///
    /// Returns [`None`] if the sampling procedure fails for any reason.
    fn sample<R>(&self, rng: &mut R) -> Option<OVector<T, D>>
    where
        R: RngExt + SeedableRng,
        DefaultAllocator: Allocator<D>;

    /// Returns an iterator that yields random samples from the distribution using the provided sampler.
    ///
    /// The iterator will yield `None` for samples that fail for any reason.
    fn sample_iter<R>(&self, rng: &mut R) -> impl Iterator<Item = Option<OVector<T, D>>>
    where
        R: RngExt + SeedableRng,
        DefaultAllocator: Allocator<D>;

    /// Returns the variance of the distribution.
    fn variance(&self) -> OVector<T, D>
    where
        DefaultAllocator: Allocator<D>;
}

// Blanket impl for mutable references.
impl<T, D, G> Density<T, D> for &G
where
    T: RealField,
    D: Dim,
    G: Density<T, D>,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, D, RStride, CStride>,
    ) -> Option<T> {
        (**self).density(sample)
    }

    fn domain(&self) -> Domain<T, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        (**self).domain()
    }

    fn log_density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, D, RStride, CStride>,
    ) -> Option<T> {
        (**self).log_density(sample)
    }

    fn mean(&self) -> OVector<T, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        (**self).mean()
    }

    fn sample<R>(&self, rng: &mut R) -> Option<OVector<T, D>>
    where
        R: RngExt + SeedableRng,
        DefaultAllocator: Allocator<D>,
    {
        (**self).sample(rng)
    }

    fn sample_iter<R>(&self, rng: &mut R) -> impl Iterator<Item = Option<OVector<T, D>>>
    where
        R: RngExt + SeedableRng,
        DefaultAllocator: Allocator<D>,
    {
        (**self).sample_iter(rng)
    }

    fn variance(&self) -> OVector<T, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        (**self).variance()
    }
}
