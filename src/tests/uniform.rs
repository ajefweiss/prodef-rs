use crate::{Density, UniformDensity};
use approx::assert_abs_diff_eq;
use nalgebra::{SVector, U1};

use super::*;

#[test]
fn test_constructor_invalid_equal_bounds() {
    let dist = UniformDensity::new(1.0, 1.0);
    assert!(dist.is_none());
}

#[test]
fn test_constructor_invalid_reversed() {
    let dist = UniformDensity::new(1.0, 0.0);
    assert!(dist.is_none());
}

#[test]
fn test_constructor_valid_standard() {
    let _dist = UniformDensity::new(0.0, 1.0).unwrap();
    // Constructor succeeded if unwrap() didn't panic
}

#[test]
fn test_constructor_valid_wide_range() {
    let _dist = UniformDensity::new(-100.0, 100.0).unwrap();
    // Constructor succeeded if unwrap() didn't panic
}

#[test]
fn test_density_at_boundaries() {
    let dist = UniformDensity::new(0.0, 1.0).unwrap();
    // Boundaries are inclusive
    assert_density_correct(&dist, 0.0, true);
    assert_density_correct(&dist, 1.0, true);
}

#[test]
fn test_density_in_domain() {
    let dist = UniformDensity::new(0.0, 1.0).unwrap();
    assert_density_correct(&dist, 0.5, true);
}

#[test]
fn test_density_outside_domain() {
    let dist = UniformDensity::new(0.0, 1.0).unwrap();
    assert_density_correct(&dist, 1.5, false);
}

#[test]
fn test_domain_bounded() {
    let dist = UniformDensity::new(0.0, 1.0).unwrap();
    let domain = dist.domain();
    let sample = SVector::from([0.5]);
    assert!(domain.contains::<U1, U1>(&sample.as_view()));
}

#[test]
fn test_sampling_containment() {
    let dist = UniformDensity::new(0.0, 1.0).unwrap();
    let samples = collect_samples(&dist, N_SAMPLES, RNG_SEED);
    assert_samples_in_bounds(&samples, 0.0, 1.0, true);
}

#[test]
fn test_sampling_determinism() {
    let dist = UniformDensity::new(0.0, 1.0).unwrap();
    assert_sampling_determinism(&dist, RNG_SEED, 100);
}

#[test]
fn test_sampling_mean_convergence() {
    let dist = UniformDensity::new(0.0, 1.0).unwrap();
    let samples = collect_samples(&dist, N_SAMPLES, RNG_SEED);
    let empirical_mean = compute_sample_mean(&samples);
    let theoretical_mean = dist.mean()[0];
    assert_abs_diff_eq!(empirical_mean, theoretical_mean, epsilon = 0.025);
}

#[test]
fn test_sampling_variance_convergence() {
    let dist = UniformDensity::new(0.0, 1.0).unwrap();
    let samples = collect_samples(&dist, N_SAMPLES, RNG_SEED);
    let mean = compute_sample_mean(&samples);
    let empirical_variance = compute_sample_variance(&samples, mean);
    let theoretical_variance = dist.variance()[0];
    assert_abs_diff_eq!(empirical_variance, theoretical_variance, epsilon = 0.025);
}
