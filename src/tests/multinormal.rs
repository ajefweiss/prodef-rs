use crate::{Density, Domain, MultivariateNormalDensity};
use approx::assert_abs_diff_eq;
use nalgebra::{Matrix2, SVector, U1, U2};
use rand::{SeedableRng, rngs::Xoshiro256PlusPlus};

use super::*;

#[test]
fn test_multinormal_constructor_identity_covariance() {
    let mean = SVector::from([0.0, 0.0]);
    let cov = Matrix2::identity();
    let domain = Domain::new_udomain(U2);
    let _dist = MultivariateNormalDensity::new(cov, domain, Some(mean)).unwrap();
}

#[test]
fn test_multinormal_sampling_containment() {
    let mean = SVector::from([0.0, 0.0]);
    let cov = Matrix2::identity();
    let domain = Domain::new_udomain(U2);
    let dist = MultivariateNormalDensity::new(cov, domain, Some(mean)).unwrap();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(RNG_SEED);
    let samples: Vec<_> = (0..N_SAMPLES)
        .filter_map(|_| dist.sample(&mut rng))
        .collect();

    // Normal distribution is unbounded, so just check we got samples
    assert_eq!(samples.len(), N_SAMPLES);
}

#[test]
fn test_multinormal_sampling_determinism() {
    let mean = SVector::from([0.0, 0.0]);
    let cov = Matrix2::identity();
    let domain = Domain::new_udomain(U2);
    let dist = MultivariateNormalDensity::new(cov, domain, Some(mean)).unwrap();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(RNG_SEED);
    let samples1: Vec<_> = (0..50).filter_map(|_| dist.sample(&mut rng)).collect();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(RNG_SEED);
    let samples2: Vec<_> = (0..50).filter_map(|_| dist.sample(&mut rng)).collect();

    assert_eq!(
        samples1, samples2,
        "Sampling not deterministic with same seed"
    );
}

#[test]
fn test_multinormal_density_at_mean() {
    let mean = SVector::from([0.0, 0.0]);
    let cov = Matrix2::identity();
    let domain = Domain::new_udomain(U2);
    let dist = MultivariateNormalDensity::new(cov, domain, Some(mean)).unwrap();

    let sample = SVector::from([0.0, 0.0]);
    let density = dist.density::<U1, U2>(&sample.as_view());

    assert!(density.is_some(), "Density should exist at mean");
    assert!(density.unwrap() > 0.0, "Density at mean should be positive");
}

#[test]
fn test_multinormal_density_shifted_mean() {
    let mean = SVector::from([5.0, -3.0]);
    let cov = Matrix2::identity();
    let domain = Domain::new_udomain(U2);
    let dist = MultivariateNormalDensity::new(cov, domain, Some(mean)).unwrap();

    // Density at mean should exist
    let density_at_mean = dist.density::<U1, U2>(&mean.as_view());
    assert!(density_at_mean.is_some(), "Density should exist at mean");

    // Density away from mean should still exist (unbounded)
    let sample_away = SVector::from([0.0, 0.0]);
    let density_away = dist.density::<U1, U2>(&sample_away.as_view());
    assert!(
        density_away.is_some(),
        "Density should exist away from mean"
    );
}

#[test]
fn test_multinormal_sampling_mean_convergence() {
    let mean = SVector::from([2.0, -1.0]);
    let cov = Matrix2::identity();
    let domain = Domain::new_udomain(U2);
    let dist = MultivariateNormalDensity::new(cov, domain, Some(mean)).unwrap();

    let samples = collect_samples_nd(&dist, N_SAMPLES, RNG_SEED);

    // Extract per-dimension samples
    let dim0_samples: Vec<f64> = samples.iter().map(|s| s[0]).collect();
    let dim1_samples: Vec<f64> = samples.iter().map(|s| s[1]).collect();

    let empirical_mean0 = compute_sample_mean(&dim0_samples);
    let empirical_mean1 = compute_sample_mean(&dim1_samples);

    // Theoretical means match the specified mean vector
    assert_abs_diff_eq!(empirical_mean0, 2.0, epsilon = 0.05);
    assert_abs_diff_eq!(empirical_mean1, -1.0, epsilon = 0.05);
}

#[test]
fn test_multinormal_sampling_variance_convergence() {
    let mean = SVector::from([0.0, 0.0]);
    let cov = Matrix2::identity();
    let domain = Domain::new_udomain(U2);
    let dist = MultivariateNormalDensity::new(cov, domain, Some(mean)).unwrap();

    let samples = collect_samples_nd(&dist, N_SAMPLES, RNG_SEED);

    // Extract per-dimension samples
    let dim0_samples: Vec<f64> = samples.iter().map(|s| s[0]).collect();
    let dim1_samples: Vec<f64> = samples.iter().map(|s| s[1]).collect();

    let dim0_mean = compute_sample_mean(&dim0_samples);
    let dim1_mean = compute_sample_mean(&dim1_samples);

    let empirical_var0 = compute_sample_variance(&dim0_samples, dim0_mean);
    let empirical_var1 = compute_sample_variance(&dim1_samples, dim1_mean);

    // Theoretical variances: Identity covariance matrix has diagonal elements of 1.0
    assert_abs_diff_eq!(empirical_var0, 1.0, epsilon = 0.1);
    assert_abs_diff_eq!(empirical_var1, 1.0, epsilon = 0.1);
}
