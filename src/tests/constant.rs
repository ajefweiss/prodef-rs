use super::*;
use crate::ConstantDensity;

#[test]
fn test_constructor_valid() {
    let dist = ConstantDensity::new(5.0);
    assert_eq!(dist.constant(), 5.0);
}

#[test]
fn test_density_at_constant() {
    let dist = ConstantDensity::new(2.5);
    assert_density_correct(&dist, 2.5, true);
}

#[test]
fn test_density_outside_constant() {
    let dist = ConstantDensity::new(2.5);
    assert_density_correct(&dist, 2.6, false);
}

#[test]
fn test_sampling_constant_value() {
    let dist = ConstantDensity::new(3.0);
    let samples = collect_samples(&dist, 100, RNG_SEED);

    for sample in samples {
        assert_eq!(sample, 3.0);
    }
}

#[test]
fn test_sampling_determinism() {
    let dist = ConstantDensity::new(7.0);
    assert_sampling_determinism(&dist, RNG_SEED, 50);
}
