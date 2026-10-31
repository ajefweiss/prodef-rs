use crate::{Density, UniformDensity, UnivariateDensity};
use approx::assert_abs_diff_eq;
use nalgebra::{SVector, U1};

use super::*;

#[test]
fn test_constructor_invalid_equal_bounds() {
    let dist = UniformDensity::new(1.0, 1.0);
    assert!(dist.is_none());
}

#[test]
fn test_construct_then_serialize_uniform() {
    // Constructor validation catches errors first
    let none_result = UniformDensity::new(1.0, 0.0); // reversed
    assert!(
        none_result.is_none(),
        "Constructor should reject reversed bounds"
    );

    // Valid construction followed by serialization
    let uniform = UniformDensity::new(0.0, 1.0).expect("valid");
    let json = serde_json::to_string(&uniform).expect("serialize");

    // Round-trip should work
    let _restored: UniformDensity<f64> = serde_json::from_str(&json).expect("roundtrip");
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
fn test_deserialize_uniform_valid_roundtrip() {
    let original = UniformDensity::new(0.0, 1.0).expect("valid constructor");
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: UniformDensity<f64> =
        serde_json::from_str(&json).expect("deserialize valid data");

    assert_eq!(restored.minimum(), original.minimum());
    assert_eq!(restored.maximum(), original.maximum());
}

#[test]
fn test_deserialize_uniform_multiple_valid_roundtrips() {
    // Test various valid ranges
    let test_cases = vec![(0.0, 1.0), (-1.0, 1.0), (-100.0, -10.0), (0.0001, 0.0002)];

    for (min, max) in test_cases {
        let original = UniformDensity::new(min, max).expect("valid constructor");
        let json = serde_json::to_string(&original).expect("serialize");
        let restored: UniformDensity<f64> =
            serde_json::from_str(&json).expect("deserialize valid data");

        assert_eq!(
            restored.minimum(),
            original.minimum(),
            "min mismatch for ({}, {})",
            min,
            max
        );
        assert_eq!(
            restored.maximum(),
            original.maximum(),
            "max mismatch for ({}, {})",
            min,
            max
        );
    }
}

#[test]
fn test_deserialize_uniform_with_unbounded_domain() {
    // Try to deserialize with UDomain (unbounded) instead of MDomain
    let invalid_json = r#"{"UDomain":1}"#;
    let result: Result<UniformDensity<f64>, _> = serde_json::from_str(invalid_json);

    assert!(result.is_err(), "Should reject unbounded domain");
}

#[test]
fn test_deserialize_uniform_with_spherical_domain() {
    // Try to deserialize with SDomain (spherical)
    let invalid_json = r#"{"SDomain":1}"#;
    let result: Result<UniformDensity<f64>, _> = serde_json::from_str(invalid_json);

    assert!(result.is_err(), "Should reject spherical domain");
}

#[test]
fn test_deserialize_univariate_density_uniform() {
    let uniform = UniformDensity::new(0.0, 1.0).expect("valid constructor");
    let uni_density = UnivariateDensity::from(uniform);
    let json = serde_json::to_string(&uni_density).expect("serialize");
    let restored: UnivariateDensity<f64> =
        serde_json::from_str(&json).expect("deserialize univariate");

    // Verify it's a Uniform variant
    let _: UniformDensity<f64> = restored.try_into().expect("should be Uniform");
}

#[test]
fn test_sampling_after_roundtrip() {
    let original = UniformDensity::new(0.0, 1.0).expect("valid");
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: UniformDensity<f64> = serde_json::from_str(&json).expect("deserialize");

    // Test that restored can be used for sampling
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let sample = restored
        .sample(&mut rng)
        .expect("sampling should work after roundtrip");

    // Verify sample is in domain
    assert!(
        sample[0] >= 0.0 && sample[0] <= 1.0,
        "Sample should be in domain after roundtrip"
    );
}

#[test]
fn test_density_evaluation_after_roundtrip() {
    let original = UniformDensity::new(0.0, 1.0).expect("valid");
    let json = serde_json::to_string(&original).expect("serialize");
    let restored: UniformDensity<f64> = serde_json::from_str(&json).expect("deserialize");

    // Test that restored can be used for density evaluation
    let sample = SVector::from([0.5]);
    let original_dens = original.density::<nalgebra::U1, nalgebra::U1>(&sample.as_view());
    let restored_dens = restored.density::<nalgebra::U1, nalgebra::U1>(&sample.as_view());

    assert_eq!(
        original_dens, restored_dens,
        "Density should match after roundtrip"
    );
}

#[test]
fn test_deserialize_invalid_json_structure() {
    // Malformed JSON
    let invalid_json = "not valid json";
    let result: Result<UniformDensity<f64>, _> = serde_json::from_str(invalid_json);

    assert!(result.is_err(), "Should reject invalid JSON");
}

#[test]
fn test_deserialize_empty_object() {
    // Empty object
    let invalid_json = "{}";
    let result: Result<UniformDensity<f64>, _> = serde_json::from_str(invalid_json);

    assert!(result.is_err(), "Should reject empty object");
}

#[test]
fn test_deserialize_wrong_type() {
    // Deserialize null instead of a domain object
    let invalid_json = "null";
    let result: Result<UniformDensity<f64>, _> = serde_json::from_str(invalid_json);

    assert!(result.is_err(), "Should reject wrong type");
}

#[test]
fn test_deserialize_multiple_errors_in_sequence() {
    // Verify that errors don't cause panics in subsequent calls
    let invalid_jsons = vec![r#"{"UDomain":1}"#, r#"{"SDomain":1}"#, r#"null"#, r#"{}"#];

    for (i, invalid_json) in invalid_jsons.iter().enumerate() {
        let result: Result<UniformDensity<f64>, _> = serde_json::from_str(invalid_json);
        assert!(
            result.is_err(),
            "Test case {} should fail: {}",
            i,
            invalid_json
        );
        // If we got here without panicking, the test passes
    }
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
