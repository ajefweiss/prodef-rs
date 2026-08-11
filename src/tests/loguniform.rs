use crate::{Density, LogUniformDensity};
use nalgebra::{SVector, U1};

use super::*;

#[test]
fn test_constructor_invalid_reversed() {
    let dist = LogUniformDensity::new(10.0, 0.1);
    assert!(dist.is_none());
}

#[test]
fn test_constructor_valid() {
    let _dist = LogUniformDensity::new(0.1, 10.0).unwrap();
    // Constructor succeeded if unwrap() didn't panic
}

#[test]
fn test_density_outside_domain() {
    let dist = LogUniformDensity::new(0.1, 10.0).unwrap();
    assert_density_correct(&dist, 0.0, false);
}

#[test]
fn test_density_within_domain() {
    let dist = LogUniformDensity::new(0.1, 10.0).unwrap();
    assert_density_correct(&dist, 1.0, true);
}

#[test]
fn test_domain_positive() {
    let dist = LogUniformDensity::new(0.1, 10.0).unwrap();
    let domain = dist.domain();
    let sample_positive = SVector::from([1.0]);
    let sample_negative = SVector::from([-1.0]);
    assert!(domain.contains::<U1, U1>(&sample_positive.as_view()));
    assert!(!domain.contains::<U1, U1>(&sample_negative.as_view()));
}

#[test]
fn test_sampling_containment() {
    let dist = LogUniformDensity::new(0.1, 10.0).unwrap();
    let samples = collect_samples(&dist, N_SAMPLES, RNG_SEED);
    assert_samples_in_bounds(&samples, 0.1, 10.0, true);
}

#[test]
fn test_sampling_determinism() {
    let dist = LogUniformDensity::new(0.1, 10.0).unwrap();
    assert_sampling_determinism(&dist, RNG_SEED, 100);
}
