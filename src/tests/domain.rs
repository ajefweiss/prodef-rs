use crate::Domain;
use nalgebra::{OVector, U1};

#[test]
fn test_boundaries_exclusive_outside() {
    // Verify that points just outside [0, 1] are NOT contained
    let domain = Domain::new_mdomain(OVector::from([(Some(0.0), Some(1.0))]));

    // Test just below lower bound (-0.001 should NOT be contained)
    let below_lower: f64 = -0.001;
    assert!(!domain.contains::<U1, U1>(&OVector::from([below_lower]).as_view()));

    // Test just above upper bound (1.001 should NOT be contained)
    let above_upper: f64 = 1.001;
    assert!(!domain.contains::<U1, U1>(&OVector::from([above_upper]).as_view()));

    // Test far outside bounds
    let far_negative: f64 = -1e6;
    assert!(!domain.contains::<U1, U1>(&OVector::from([far_negative]).as_view()));

    let far_positive: f64 = 1e6;
    assert!(!domain.contains::<U1, U1>(&OVector::from([far_positive]).as_view()));
}

#[test]
fn test_boundaries_inclusive() {
    // Verify that boundary points [0, 1] are inclusive in bounded domain
    let domain = Domain::new_mdomain(OVector::from([(Some(0.0), Some(1.0))]));

    // Test lower boundary (0.0 should be contained)
    let lower_bound: f64 = 0.0;
    assert!(domain.contains::<U1, U1>(&OVector::from([lower_bound]).as_view()));

    // Test upper boundary (1.0 should be contained)
    let upper_bound: f64 = 1.0;
    assert!(domain.contains::<U1, U1>(&OVector::from([upper_bound]).as_view()));

    // Test interior point (0.5 should be contained)
    let interior: f64 = 0.5;
    assert!(domain.contains::<U1, U1>(&OVector::from([interior]).as_view()));
}

#[test]
fn test_clamp_above_maximum() {
    let domain = Domain::new_mdomain(OVector::from([(Some(0.0), Some(1.0))]));
    let sample_above = OVector::from([1.5]);
    let clamped = domain.clamp::<U1, U1>(&sample_above.as_view());
    assert_eq!(clamped[0], 1.0);
}

#[test]
fn test_clamp_below_minimum() {
    let domain = Domain::new_mdomain(OVector::from([(Some(0.0), Some(1.0))]));
    let sample_below = OVector::from([-0.5]);
    let clamped = domain.clamp::<U1, U1>(&sample_below.as_view());
    assert_eq!(clamped[0], 0.0);
}

#[test]
fn test_clamp_half_bounded_lower() {
    // Test clamping with only lower bound (no upper bound)
    let domain = Domain::new_mdomain(OVector::from([(Some(0.0), None)]));

    // Value below lower bound should be clamped to lower bound
    let below = OVector::from([-1.0]);
    let clamped = domain.clamp::<U1, U1>(&below.as_view());
    assert_eq!(clamped[0], 0.0);

    // Value above lower bound should remain unchanged (no upper bound)
    let above = OVector::from([1e6]);
    let clamped = domain.clamp::<U1, U1>(&above.as_view());
    assert_eq!(clamped[0], 1e6);
}

#[test]
fn test_clamp_half_bounded_upper() {
    // Test clamping with only upper bound (no lower bound)
    let domain = Domain::new_mdomain(OVector::from([(None, Some(1.0))]));

    // Value above upper bound should be clamped to upper bound
    let above = OVector::from([2.0]);
    let clamped = domain.clamp::<U1, U1>(&above.as_view());
    assert_eq!(clamped[0], 1.0);

    // Value below upper bound should remain unchanged (no lower bound)
    let below = OVector::from([-1e6]);
    let clamped = domain.clamp::<U1, U1>(&below.as_view());
    assert_eq!(clamped[0], -1e6);
}

#[test]
fn test_clamp_unbounded() {
    // Test that clamping on unbounded domain returns sample unchanged
    let domain = Domain::new_udomain(U1);

    let sample = OVector::from([42.0]);
    let clamped = domain.clamp::<U1, U1>(&sample.as_view());
    assert_eq!(clamped[0], 42.0);
}

#[test]
fn test_clamp_with_explicit_bounds() {
    // Test clamping with explicit lower and upper bounds
    let domain = Domain::new_mdomain(OVector::from([(Some(0.0), Some(1.0))]));

    // Value below lower bound should be clamped to lower bound
    let below = OVector::from([-0.5]);
    let clamped = domain.clamp::<U1, U1>(&below.as_view());
    assert_eq!(clamped[0], 0.0);

    // Value above upper bound should be clamped to upper bound
    let above = OVector::from([1.5]);
    let clamped = domain.clamp::<U1, U1>(&above.as_view());
    assert_eq!(clamped[0], 1.0);

    // Value within bounds should remain unchanged
    let inside = OVector::from([0.5]);
    let clamped = domain.clamp::<U1, U1>(&inside.as_view());
    assert_eq!(clamped[0], 0.5);
}

#[test]
fn test_contains_unbounded() {
    let domain: Domain<f64, U1> = Domain::UDomain(U1);
    let sample = OVector::from([-1e6]);
    assert!(domain.contains::<U1, U1>(&sample.as_view()));
}

#[test]
fn test_half_bounded_lower() {
    // Verify half-bounded domains work correctly (lower bound only, upper unbounded)
    // Note: lower bound is inclusive (value >= min)
    let domain = Domain::new_mdomain(OVector::from([(Some(0.0), None)]));

    // Exactly at lower bound should be contained (inclusive)
    assert!(domain.contains::<U1, U1>(&OVector::from([0.0]).as_view()));

    // Just above lower bound should be contained
    assert!(domain.contains::<U1, U1>(&OVector::from([1e-9]).as_view()));

    // Inside should be contained
    assert!(domain.contains::<U1, U1>(&OVector::from([1e6]).as_view()));

    // Below lower bound should NOT be contained
    assert!(!domain.contains::<U1, U1>(&OVector::from([-1e-9]).as_view()));
}

#[test]
fn test_half_bounded_upper() {
    // Verify half-bounded domains work correctly (upper bound only, lower unbounded)
    // Note: upper bound is inclusive (value <= max)
    let domain = Domain::new_mdomain(OVector::from([(None, Some(1.0))]));

    // Exactly at upper bound should be contained (inclusive)
    assert!(domain.contains::<U1, U1>(&OVector::from([1.0]).as_view()));

    // Just below upper bound should be contained
    assert!(domain.contains::<U1, U1>(&OVector::from([1.0 - 1e-9]).as_view()));

    // Inside should be contained
    assert!(domain.contains::<U1, U1>(&OVector::from([-1e6]).as_view()));

    // Above upper bound should NOT be contained
    assert!(!domain.contains::<U1, U1>(&OVector::from([1.0 + 1e-9]).as_view()));
}

#[test]
fn test_maximum_values() {
    let domain = Domain::new_mdomain(OVector::from([(Some(0.0), Some(1.0)), (Some(-5.0), None)]));
    let maxes = domain.maximum_values();
    assert_eq!(maxes[0], Some(1.0));
    assert_eq!(maxes[1], None);
}

#[test]
fn test_spherical_domain_contains() {
    use nalgebra::U2;

    let domain: Domain<f64, U2> = Domain::new_sdomain(U2);

    // Test center (origin)
    assert!(domain.contains::<U1, U2>(&OVector::from([0.0, 0.0]).as_view()));

    // Test on boundary (radius = 1)
    assert!(domain.contains::<U1, U2>(&OVector::from([1.0, 0.0]).as_view()));
    assert!(domain.contains::<U1, U2>(&OVector::from([0.0, 1.0]).as_view()));

    // Test interior point
    assert!(domain.contains::<U1, U2>(&OVector::from([0.6, 0.8]).as_view()));

    // Test outside
    assert!(!domain.contains::<U1, U2>(&OVector::from([1.1, 0.0]).as_view()));
    assert!(!domain.contains::<U1, U2>(&OVector::from([0.8, 0.8]).as_view()));
}

#[test]
fn test_spherical_domain_clamp() {
    use nalgebra::U2;

    let domain: Domain<f64, U2> = Domain::new_sdomain(U2);

    // Test point inside (should remain unchanged)
    let inside = OVector::from([0.5, 0.5]);
    let clamped = domain.clamp::<U1, U2>(&inside.as_view());
    assert!((clamped[0] - 0.5).abs() < 1e-9);
    assert!((clamped[1] - 0.5).abs() < 1e-9);

    // Test point outside (should be projected to boundary)
    let outside = OVector::from([2.0, 0.0]);
    let clamped = domain.clamp::<U1, U2>(&outside.as_view());
    assert!((clamped[0] - 1.0).abs() < 1e-9);
    assert!((clamped[1] - 0.0).abs() < 1e-9);

    // Test point on boundary (should remain unchanged)
    let boundary = OVector::from([0.6, 0.8]);
    let clamped = domain.clamp::<U1, U2>(&boundary.as_view());
    assert!((clamped[0] - 0.6).abs() < 1e-9);
    assert!((clamped[1] - 0.8).abs() < 1e-9);
}

#[test]
fn test_spherical_shell_domain_contains() {
    use nalgebra::U2;

    let domain: Domain<f64, U2> = Domain::new_shdomain(U2);

    // Test on boundary (radius = 1) - should be contained with tolerance
    assert!(domain.contains::<U1, U2>(&OVector::from([1.0, 0.0]).as_view()));
    assert!(domain.contains::<U1, U2>(&OVector::from([0.0, 1.0]).as_view()));
    assert!(domain.contains::<U1, U2>(&OVector::from([0.6, 0.8]).as_view()));

    // Test center (origin) - should NOT be contained
    assert!(!domain.contains::<U1, U2>(&OVector::from([0.0, 0.0]).as_view()));

    // Test interior point - should NOT be contained
    assert!(!domain.contains::<U1, U2>(&OVector::from([0.5, 0.5]).as_view()));

    // Test outside - should NOT be contained
    assert!(!domain.contains::<U1, U2>(&OVector::from([1.1, 0.0]).as_view()));
    assert!(!domain.contains::<U1, U2>(&OVector::from([0.8, 0.8]).as_view()));
}

#[test]
fn test_spherical_shell_domain_clamp() {
    use nalgebra::U2;

    let domain: Domain<f64, U2> = Domain::new_shdomain(U2);

    // Test point inside (should be projected to boundary)
    let inside = OVector::from([0.5, 0.5]);
    let clamped = domain.clamp::<U1, U2>(&inside.as_view());
    let norm = (clamped[0] * clamped[0] + clamped[1] * clamped[1]).sqrt();
    assert!((norm - 1.0).abs() < 1e-9);

    // Test point outside (should be projected to boundary)
    let outside = OVector::from([2.0, 0.0]);
    let clamped = domain.clamp::<U1, U2>(&outside.as_view());
    assert!((clamped[0] - 1.0).abs() < 1e-9);
    assert!((clamped[1] - 0.0).abs() < 1e-9);

    // Test point on boundary (should remain unchanged)
    let boundary = OVector::from([0.6, 0.8]);
    let clamped = domain.clamp::<U1, U2>(&boundary.as_view());
    assert!((clamped[0] - 0.6).abs() < 1e-9);
    assert!((clamped[1] - 0.8).abs() < 1e-9);
}

#[test]
fn test_spherical_shell_domain_zero_vector() {
    use nalgebra::U3;

    let domain: Domain<f64, U3> = Domain::new_shdomain(U3);

    // Test zero vector (should be clamped to (1, 0, 0))
    let zero = OVector::from([0.0, 0.0, 0.0]);
    let clamped = domain.clamp::<U1, U3>(&zero.as_view());
    assert!((clamped[0] - 1.0).abs() < 1e-9);
    assert!((clamped[1] - 0.0).abs() < 1e-9);
    assert!((clamped[2] - 0.0).abs() < 1e-9);
}
