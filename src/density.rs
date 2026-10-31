use crate::Domain;
use nalgebra::{DefaultAllocator, Dim, OVector, RealField, VectorView, allocator::Allocator};
use rand::{RngExt, SeedableRng};

/// A trait that is shared by all probability density functions.
pub trait Density<T: RealField, D: Dim>: Clone {
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
        R: RngExt,
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
        R: RngExt,
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
