#![doc = include_str!("../README.md")]

mod domain;
mod macros;
mod multinormal;
mod multivariate;
mod particle;
mod univariate;

// Enable access to the Python types if the "pyo3" feature is enabled.
#[cfg(feature = "pyo3")]
pub mod pytypes;

#[cfg(test)]
mod tests;

pub use domain::Domain;
pub use multinormal::MultivariateNormalDensity;
pub use multivariate::MultivariateDensity;
pub use particle::ParticleDensity;
pub use univariate::{
    ConstantDensity, CosineDensity, LogUniformDensity, LognormalDensity, NormalDensity,
    UniformDensity, UnivariateDensity,
};

use nalgebra::{DefaultAllocator, Dim, OVector, RealField, VectorView, allocator::Allocator};
use rand::RngExt;
use serde::{Deserialize, Serialize};

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
/// **Sampling**: The `sample()` method uses a configurable `SamplingMode` enum to handle
/// boundary constraints during sampling:
/// - `SingleAttempt`: Fail fast if out-of-domain
/// - `UntilValid`: Retry with budget limit (default: 512 attempts)
/// - `UntilValidOrClamp`: Clamp to boundary if attempts budget exceeded
/// - `UntilValidNoLimit`: Keep resampling until valid
///
/// The `sample_iter()` method provides an iterator interface for generating samples according to the specified sampling mode, yielding `None` for failed sampling attempts.
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
/// # use prodef::{Density, SamplingMode};
/// # use nalgebra::{SVector, U1};
/// # use rand::{SeedableRng, rngs::StdRng};
/// # let normal = prodef::NormalDensity::new(0.0, 1.0, Some(-3.0), Some(3.0)).unwrap();
/// let mut rng = StdRng::seed_from_u64(42);
/// let mode = SamplingMode::default();
///
/// if let Some(sample) = (&normal).sample(&mut rng, &mode) {
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
    /// Calculates, or estimates, a density value for a sample.
    /// Returns [`None`] if the sample is outside of the function domain.
    ///
    /// Note that the returned value is not necessarily normalized.
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, D, RStride, CStride>,
    ) -> Option<T>;

    /// Returns the underlying function [`Domain`].
    fn domain(&self) -> Domain<T, D>
    where
        DefaultAllocator: Allocator<D>;

    /// Returns the mean of the distribution.
    fn mean(&self) -> OVector<T, D>
    where
        DefaultAllocator: Allocator<D>;

    /// Returns the number of dimensions of the distribution.
    fn ndims(&self) -> usize where DefaultAllocator: Allocator<D> {
        self.domain().shape_generic().value()
    }

    /// Draw a random sample from the probability density distribution using the provided random number generator and sampling mode.
    ///
    /// Returns [`None`] if the sampling procedure fails (too many attempted draws).
    fn sample(&self, rng: &mut impl RngExt, mode: &SamplingMode) -> Option<OVector<T, D>>
    where
        DefaultAllocator: Allocator<D>;

    /// Returns an iterator that yields random samples from the distribution according to the specified sampling mode.
    ///
    /// This function behaves the same as [`Density::sample`] when using [`SamplingMode::SingleAttempt`].
    /// The iterator will yield `None` for samples that fail the sampling procedure (e.g., due to too many attempts in rejection sampling).
    fn sample_iter(&self, rng: &mut impl RngExt) -> impl Iterator<Item = Option<OVector<T, D>>>
    where
        DefaultAllocator: Allocator<D>;

        
    /// Returns the variance of the distribution.
    fn variance(&self) -> OVector<T, D>
    where
        DefaultAllocator: Allocator<D>;
}

// Blanket impl for all references.
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

    fn mean(&self) -> OVector<T, D>
    where
        DefaultAllocator: Allocator<D>,
    {
        (**self).mean()
    }

    fn sample(&self, rng: &mut impl RngExt, mode: &SamplingMode) -> Option<OVector<T, D>>
    where
        DefaultAllocator: Allocator<D>,
    {
        (**self).sample(rng, mode)
    }

    fn sample_iter(&self, rng: &mut impl RngExt) -> impl Iterator<Item = Option<OVector<T, D>>>
    where
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

/// Helper for implementing acceptance-rejection sampling with configurable modes.
pub trait RejectionSampler<T, D>: Density<T, D>
where
    T: RealField,
    D: Dim,
{
    /// Generate a single candidate sample
    fn generate_candidate(&self, rng: &mut impl RngExt) -> OVector<T, D>
    where
        DefaultAllocator: Allocator<D>;

    /// Execute rejection sampling with mode handling
    fn rejection_sample(&self, rng: &mut impl RngExt, mode: &SamplingMode) -> Option<OVector<T, D>>
    where
        DefaultAllocator: Allocator<D>,
    {
        let mut attempts = 0;

        loop {
            let candidate = self.generate_candidate(rng);

            if self.domain().contains(&candidate.as_view()) {
                return Some(candidate);
            }

            match mode {
                SamplingMode::SingleAttempt => return None,
                SamplingMode::UntilValid { max_attempts } => {
                    if attempts >= *max_attempts {
                        return None;
                    }
                    attempts += 1;
                }
                SamplingMode::UntilValidNoLimit => {
                    attempts += 1;
                }
                SamplingMode::UntilValidOrClamp { max_attempts } => {
                    if attempts >= *max_attempts {
                        return Some(self.domain().clamp(&candidate.as_view()));
                    }
                    attempts += 1;
                }
            }
        }
    }
}

/// Sampling mode for the [`Density::sample`] function.
///
/// `SamplingMode` controls the behavior of rejection sampling when generating samples
/// from bounded or constrained distributions.
///
/// # Variants
///
/// - **`SingleAttempt`**: Make one draw; return `None` if invalid.
/// - **`UntilValid { max_attempts }`**: Retry rejection sampling up to `max_attempts` times.
///   Returns `None` if all attempts produce invalid samples. Default: 512 attempts.
/// - **`UntilValidOrClamp { max_attempts }`**: Retry rejection sampling up to `max_attempts`
///   times; clamp to domain boundary if budget exhausted.
/// - **`UntilValidNoLimit`**: Retry rejection sampling indefinitely until a valid sample
///   is found. **Caution**: May loop infinitely if acceptance region is empty or vanishingly small.
///
/// # Examples
///
/// Single attempt mode:
/// ```
/// # use prodef::{Density, SamplingMode};
/// # use rand::{SeedableRng, rngs::StdRng};
/// # let normal = prodef::NormalDensity::new(0.0, 1.0, Some(-1.0), Some(1.0)).unwrap();
/// let mut rng = StdRng::seed_from_u64(42);
/// let sample = (&normal).sample(&mut rng, &SamplingMode::SingleAttempt);
/// ```
///
/// Bounded retries (default):
/// ```
/// # use prodef::{Density, SamplingMode};
/// # use rand::{SeedableRng, rngs::StdRng};
/// # let normal = prodef::NormalDensity::new(0.0, 1.0, Some(-3.0), Some(3.0)).unwrap();
/// let mut rng = StdRng::seed_from_u64(42);
/// let sample = (&normal).sample(&mut rng, &SamplingMode::default());
/// ```
///
/// Clamp on failure:
/// ```
/// # use prodef::{Density, SamplingMode};
/// # use rand::{SeedableRng, rngs::StdRng};
/// # let normal = prodef::NormalDensity::new(0.0, 1.0, Some(0.0), Some(1.0)).unwrap();
/// let mut rng = StdRng::seed_from_u64(42);
/// let mode = SamplingMode::UntilValidOrClamp { max_attempts: 50 };
/// let sample = (&normal).sample(&mut rng, &mode);
/// ```
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub enum SamplingMode {
    /// Sample once; return `None` if invalid. No retries.
    SingleAttempt,
    /// Retry rejection sampling up to `max_attempts` times, returning `None` if all fail.
    UntilValid {
        /// Maximum number of sampling attempts.
        max_attempts: usize,
    },
    /// Retry rejection sampling up to `max_attempts` times; clamp to domain boundary if exhausted.
    UntilValidOrClamp {
        /// Maximum number of sampling attempts.
        max_attempts: usize,
    },
    /// Retry rejection sampling indefinitely until a valid sample is found.
    /// **Caution**: May loop infinitely if acceptance region is empty or vanishingly small.
    UntilValidNoLimit,
}

impl Default for SamplingMode {
    fn default() -> Self {
        SamplingMode::UntilValid { max_attempts: 512 }
    }
}
