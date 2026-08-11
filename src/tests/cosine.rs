use crate::{
    CosineDensity, Density,
    tests::{
        N_SAMPLES, RNG_SEED, assert_density_correct, assert_samples_in_bounds,
        assert_sampling_determinism, collect_samples,
    },
};
use nalgebra::{SVector, U1};

#[test]
fn test_constructor_invalid_reversed() {
    let dist = CosineDensity::new(0.2, 0.1);
    assert!(dist.is_none());
}

#[test]
fn test_constructor_valid() {
    let _dist = CosineDensity::new(0.1, 0.2).unwrap();
    // Constructor succeeded if unwrap() didn't panic
}

#[test]
fn test_density_outside_domain() {
    let dist = CosineDensity::new(0.1, 0.2).unwrap();
    assert_density_correct(&dist, 0.0, false);
}

#[test]
fn test_density_within_domain() {
    let dist = CosineDensity::new(0.1, 0.2).unwrap();
    assert_density_correct(&dist, 0.15, true);
}

#[test]
fn test_domain_bounded() {
    let dist = CosineDensity::new(0.1, 0.2).unwrap();
    let domain = dist.domain();
    let sample_in = SVector::from([0.15]);
    let sample_out = SVector::from([0.5]);
    assert!(domain.contains::<U1, U1>(&sample_in.as_view()));
    assert!(!domain.contains::<U1, U1>(&sample_out.as_view()));
}

#[test]
fn test_sampling_containment() {
    let dist = CosineDensity::new(0.1, 0.2).unwrap();
    let samples = collect_samples(&dist, N_SAMPLES, RNG_SEED);
    assert_samples_in_bounds(&samples, 0.1, 0.2, true);
}

#[test]
fn test_sampling_determinism() {
    let dist = CosineDensity::new(0.1, 0.2).unwrap();
    assert_sampling_determinism(&dist, RNG_SEED, 100);
}
