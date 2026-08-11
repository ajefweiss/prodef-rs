//! Comprehensive testing module for ProDeF distributions.
//!
//! All tests for probability distributions are contained in this module.
//! Tests follow a consistent pattern: constructor validation, sampling behavior,
//! density evaluation, and domain constraint checking.
//!
//! # Test Organization
//!
//! - **Constructor Tests**: Validate parameter acceptance/rejection
//! - **Sampling Tests**: Check determinism, convergence, and domain containment
//! - **Density Tests**: Verify PDF evaluation and None handling
//! - **Domain Tests**: Confirm boundary behavior and containment
//!
//! # Distributions Tested
//!
//! **Univariate:**
//! - Constant: Fixed value
//! - Uniform: Bounded uniform
//! - Normal: Gaussian (unbounded and bounded)
//! - Cosine: Cosine-shaped
//! - Lognormal: Log-normal
//! - LogUniform: Log-uniform
//!
//! **Multivariate:**
//! - MultivariateDensity: Product of independent univariates
//! - MultivariateNormalDensity: Gaussian with full covariance matrix
//! - ParticleDensity: Kernel density estimation from weighted particles

mod constant;
mod conversions;
mod cosine;
// mod kent;
mod lognormal;
mod loguniform;
mod multinormal;
mod multivariate;
mod normal;
mod particle;
mod uniform;

use crate::{Density, tval};
use nalgebra::{DefaultAllocator, OVector, RealField, SVector, U1, allocator::Allocator};
use rand::{SeedableRng, rngs::Xoshiro256PlusPlus};

/// Large sample size for statistical convergence tests
const N_SAMPLES: usize = 10_000;

/// Default RNG seed for reproducible tests
const RNG_SEED: u64 = 42;

/// Asserts density evaluation within and outside domain.
fn assert_density_correct<'a, D>(dist: &'a D, sample: f64, should_exist: bool)
where
    &'a D: Density<f64, U1>,
{
    let sample_vec = SVector::from([sample]);
    let density = dist.density::<U1, U1>(&sample_vec.as_view());

    if should_exist {
        assert!(density.is_some(), "Expected density to exist at {}", sample);
        assert!(density.unwrap() > 0.0, "Density should be positive");
    } else {
        assert!(
            density.is_none(),
            "Expected density to be None at {}",
            sample
        );
    }
}

/// Asserts all samples fall within bounds.
fn assert_samples_in_bounds<T: PartialOrd + std::fmt::Display>(
    samples: &[T],
    min: T,
    max: T,
    inclusive: bool,
) {
    for sample in samples {
        let in_bounds = if inclusive {
            sample >= &min && sample <= &max
        } else {
            sample > &min && sample < &max
        };
        assert!(
            in_bounds,
            "Sample {} outside bounds [{}, {}]",
            sample, min, max
        );
    }
}

/// Asserts sampling determinism: same seed produces identical samples.
fn assert_sampling_determinism<'a, D, T, Dim>(dist: &'a D, seed: u64, n_samples: usize)
where
    T: RealField + PartialEq + std::fmt::Debug,
    Dim: nalgebra::Dim,
    DefaultAllocator: Allocator<Dim>,
    &'a D: Density<T, Dim>,
{
    let samples1 = collect_samples_nd(dist, n_samples, seed);
    let samples2 = collect_samples_nd(dist, n_samples, seed);

    assert_eq!(
        samples1, samples2,
        "Sampling not deterministic with same seed"
    );
}

/// Collects N samples from an N-dimensional distribution with deterministic seeding.
/// Returns Vec<OVector<T, D>> where OVector contains the D-dimensional samples.
/// This is the canonical collection function; use it for all distributions.
fn collect_samples_nd<'a, T, D, G>(dist: &'a G, n_samples: usize, seed: u64) -> Vec<OVector<T, D>>
where
    T: RealField,
    D: nalgebra::Dim,
    &'a G: Density<T, D>,
    DefaultAllocator: Allocator<D>,
{
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    (0..n_samples)
        .filter_map(|_| dist.sample(&mut rng))
        .collect()
}

/// Convenience wrapper: Collects N samples from a 1D distribution and extracts scalar values.
/// Uses collect_samples_nd internally. Returns Vec<T> where T is the scalar type.
fn collect_samples<'a, T, D>(dist: &'a D, n_samples: usize, seed: u64) -> Vec<T>
where
    T: RealField,
    &'a D: Density<T, U1>,
{
    let samples = collect_samples_nd(dist, n_samples, seed);
    samples.into_iter().map(|s| s[0].clone()).collect()
}

/// Computes the mean of a vector of samples.
fn compute_sample_mean<T: RealField>(samples: &[T]) -> T {
    if samples.is_empty() {
        return T::zero();
    }

    let sum = samples.iter().cloned().fold(T::zero(), T::add);
    sum / tval!(samples.len(), usize)
}

/// Computes the sample variance from samples and mean.
fn compute_sample_variance<T: RealField>(samples: &[T], mean: T) -> T {
    if samples.len() < 2 {
        return T::zero();
    }

    let sum_sq_diff = samples
        .iter()
        .cloned()
        .fold(T::zero(), |acc, x| acc + (x - mean.clone()).powi(2));

    sum_sq_diff / tval!(samples.len() - 1, usize)
}

/// Computes the mean vector of N-D samples.
/// For D-dimensional samples, returns the arithmetic mean across all samples.
#[allow(dead_code)]
fn compute_sample_mean_nd<T: RealField, D: nalgebra::Dim>(
    samples: &[OVector<T, D>],
) -> OVector<T, D>
where
    DefaultAllocator: Allocator<D>,
{
    if samples.is_empty() {
        return OVector::zeros_generic(samples[0].shape_generic().0, U1);
    }

    let sum = samples.iter().cloned().fold(
        OVector::zeros_generic(samples[0].shape_generic().0, U1),
        |acc, x| acc + x,
    );
    sum / tval!(samples.len(), usize)
}

/// Extracts a single dimension from N-D samples.
/// Returns a vector of scalar values for the specified dimension index.
#[allow(dead_code)]
fn extract_dimension<T: RealField, D: nalgebra::Dim>(
    samples: &[OVector<T, D>],
    dim: usize,
) -> Vec<T>
where
    DefaultAllocator: Allocator<D>,
{
    samples.iter().map(|s| s[dim].clone()).collect()
}
