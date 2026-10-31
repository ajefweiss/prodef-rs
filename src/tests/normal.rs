use crate::{Density, NormalDensity, UnivariateDensity};
use approx::assert_abs_diff_eq;
use nalgebra::{SVector, U1};

use super::*;

#[test]
fn test_construct_then_serialize_normal() {
    // Constructor validation catches errors first
    let none_result = NormalDensity::new(0.0, 0.0, None, None); // zero std_dev
    assert!(
        none_result.is_none(),
        "Constructor should reject zero std_dev"
    );

    let none_result = NormalDensity::new(0.0, 1.0, Some(1.0), Some(0.0)); // reversed bounds
    assert!(
        none_result.is_none(),
        "Constructor should reject reversed bounds"
    );

    // Valid construction followed by serialization
    let normal = NormalDensity::new(0.0, 1.0, Some(-2.0), Some(2.0)).expect("valid");
    let json = serde_json::to_string(&normal).expect("serialize");

    // Round-trip should work
    let _restored: NormalDensity<f64> = serde_json::from_str(&json).expect("roundtrip");
}

#[test]
fn test_constructor_invalid_negative_variance() {
    let dist = NormalDensity::new(0.0, -1.0, None, None);
    assert!(dist.is_none());
}

#[test]
fn test_constructor_invalid_zero_variance() {
    let dist = NormalDensity::new(0.0, 0.0, None, None);
    assert!(dist.is_none());
}

#[test]
fn test_constructor_valid_shifted() {
    let _dist = NormalDensity::new(3.0, 2.0, None, None).unwrap();
    // Constructor succeeded if unwrap() didn't panic
}

#[test]
fn test_constructor_valid_standard() {
    let _dist = NormalDensity::new(0.0, 1.0, None, None).unwrap();
    // Constructor succeeded if unwrap() didn't panic
}

#[test]
fn test_density_at_mean() {
    let dist = NormalDensity::new(0.0, 1.0, None, None).unwrap();
    assert_density_correct(&dist, 0.0, true);
}

#[test]
fn test_density_away_from_mean() {
    let dist = NormalDensity::new(0.0, 1.0, None, None).unwrap();
    assert_density_correct(&dist, 2.0, true);
}

#[test]
fn test_domain_unbounded() {
    let dist = NormalDensity::new(0.0, 1.0, None, None).unwrap();
    let domain = dist.domain();
    // Should accept both small and large values
    let sample_small = SVector::from([-1000.0]);
    let sample_large = SVector::from([1000.0]);
    assert!(domain.contains::<U1, U1>(&sample_small.as_view()));
    assert!(domain.contains::<U1, U1>(&sample_large.as_view()));
}

#[test]
fn test_deserialize_normal_valid_roundtrip() {
    let original = NormalDensity::new(0.0, 1.0, None, None).expect("valid constructor");
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: NormalDensity<f64> = serde_json::from_str(&json).expect("deserialize valid data");

    // Verify the domain is preserved
    assert_eq!(restored.minimum(), original.minimum());
    assert_eq!(restored.maximum(), original.maximum());
}

#[test]
fn test_deserialize_normal_with_bounded_valid_roundtrip() {
    let original =
        NormalDensity::new(0.0, 1.0, Some(-3.0), Some(3.0)).expect("valid bounded constructor");
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: NormalDensity<f64> = serde_json::from_str(&json).expect("deserialize valid data");

    // Verify the bounds and domain are preserved
    assert_eq!(restored.minimum(), original.minimum());
    assert_eq!(restored.maximum(), original.maximum());
}

#[test]
fn test_deserialize_normal_multiple_valid_roundtrips() {
    // Test various valid parameters
    let test_cases = vec![
        (0.0, 1.0, None, None),
        (0.0, 1.0, Some(-1.0), Some(1.0)),
        (10.0, 2.0, Some(0.0), Some(20.0)),
        (-5.0, 0.5, Some(-6.0), Some(-4.0)),
    ];

    for (mean, std_dev, lower, upper) in test_cases {
        let original = NormalDensity::new(mean, std_dev, lower, upper).expect("valid constructor");
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: NormalDensity<f64> =
            serde_json::from_str(&json).expect("deserialize valid data");

        assert_eq!(
            restored.minimum(),
            original.minimum(),
            "min mismatch for ({}, {}, {:?}, {:?})",
            mean,
            std_dev,
            lower,
            upper
        );
        assert_eq!(
            restored.maximum(),
            original.maximum(),
            "max mismatch for ({}, {}, {:?}, {:?})",
            mean,
            std_dev,
            lower,
            upper
        );
    }
}

#[test]
fn test_deserialize_univariate_density_normal() {
    let normal = NormalDensity::new(0.5, 0.1, Some(0.0), Some(1.0)).expect("valid constructor");
    let uni_density = UnivariateDensity::from(normal);
    let json = serde_json::to_string(&uni_density).expect("serialize");
    let restored: UnivariateDensity<f64> =
        serde_json::from_str(&json).expect("deserialize univariate");

    // Verify it's a Normal variant
    let _: NormalDensity<f64> = restored.try_into().expect("should be Normal");
}

#[test]
fn test_unbounded_sampling_determinism() {
    let dist = NormalDensity::new(0.0, 1.0, None, None).unwrap();
    assert_sampling_determinism(&dist, RNG_SEED, 100);
}

#[test]
fn test_unbounded_sampling_mean_convergence() {
    let dist = NormalDensity::new(0.0, 1.0, None, None).unwrap();
    let samples = collect_samples(&dist, N_SAMPLES, RNG_SEED);
    let empirical_mean = compute_sample_mean(&samples);
    let theoretical_mean = dist.mean()[0];
    assert_abs_diff_eq!(empirical_mean, theoretical_mean, epsilon = 0.08);
}

#[test]
fn test_unbounded_sampling_variance_convergence() {
    let dist = NormalDensity::new(0.0, 1.0, None, None).unwrap();
    let samples = collect_samples(&dist, N_SAMPLES, RNG_SEED);
    let mean = compute_sample_mean(&samples);
    let empirical_variance = compute_sample_variance(&samples, mean);
    let theoretical_variance = dist.variance()[0];
    assert_abs_diff_eq!(empirical_variance, theoretical_variance, epsilon = 0.08);
}

#[test]
fn test_bounded_sampling_asymmetric_containment() {
    // μ=5, σ=1, bounds=[0, 1] (heavily truncated)
    let dist = NormalDensity::new(3.0, 1.0, Some(0.0), Some(1.0)).unwrap();
    let samples = collect_samples(&dist, N_SAMPLES, RNG_SEED);
    assert_samples_in_bounds(&samples, 0.0, 1.0, true);
}

#[test]
fn test_bounded_sampling_asymmetric_determinism() {
    let dist = NormalDensity::new(2.0, 1.0, Some(0.0), Some(1.0)).unwrap();
    assert_sampling_determinism(&dist, RNG_SEED, 100);
}

#[test]
fn test_bounded_sampling_symmetric_containment() {
    let dist = NormalDensity::new(0.0, 1.0, Some(-3.0), Some(3.0)).unwrap();
    let samples = collect_samples(&dist, N_SAMPLES, RNG_SEED);
    assert_samples_in_bounds(&samples, -3.0, 3.0, true);
}

#[test]
fn test_bounded_sampling_symmetric_determinism() {
    let dist = NormalDensity::new(0.0, 1.0, Some(-3.0), Some(3.0)).unwrap();
    assert_sampling_determinism(&dist, RNG_SEED, 100);
}

#[test]
fn test_bounded_sampling_symmetric_mean() {
    let dist = NormalDensity::new(0.0, 1.0, Some(-3.0), Some(3.0)).unwrap();
    let samples = collect_samples(&dist, N_SAMPLES, RNG_SEED);
    let empirical_mean = compute_sample_mean(&samples);
    let theoretical_mean = dist.mean()[0];
    assert_abs_diff_eq!(empirical_mean, theoretical_mean, epsilon = 0.1);
}

#[test]
fn test_density_outside_bounds() {
    let dist = NormalDensity::new(0.0, 1.0, Some(-2.0), Some(2.0)).unwrap();
    assert_density_correct(&dist, 3.0, false);
}

#[test]
fn test_density_within_bounds() {
    let dist = NormalDensity::new(0.0, 1.0, Some(-2.0), Some(2.0)).unwrap();
    assert_density_correct(&dist, 0.0, true);
}

#[test]
fn test_domain_bounded_inclusive() {
    let dist = NormalDensity::new(0.0, 1.0, Some(-2.0), Some(2.0)).unwrap();
    let domain = dist.domain();
    // Boundaries should be inclusive
    let at_min = SVector::from([-2.0]);
    let at_max = SVector::from([2.0]);
    assert!(domain.contains::<U1, U1>(&at_min.as_view()));
    assert!(domain.contains::<U1, U1>(&at_max.as_view()));
}
