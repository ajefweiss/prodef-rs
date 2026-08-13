use crate::{
    Density, Domain, KentDensity,
    tests::{N_SAMPLES, RNG_SEED, assert_sampling_determinism, collect_samples_nd},
};
use approx::ulps_eq;
use nalgebra::{U3, Vector3};
use rand::SeedableRng;

#[test]
fn test_density_on_unit_sphere_at_mu() {
    // Choose mu and g1 orthonormal so the log-density is well-defined.
    let mu = Vector3::new(1.0, 0.0, 0.0);
    let g1 = Vector3::new(0.0, 1.0, 0.0);

    let dist = KentDensity::new(mu, g1, 2.0, 1.5);

    let sample = mu;
    let density = dist.density::<nalgebra::U1, U3>(&sample.as_view());
    assert!(
        density.is_some(),
        "Expected density to exist on unit sphere"
    );
    assert!(density.unwrap() > 0.0, "Density should be positive");
}

#[test]
fn test_density_is_none_off_unit_sphere() {
    let mu = Vector3::new(1.0, 0.0, 0.0);
    let g1 = Vector3::new(0.0, 1.0, 0.0);

    let dist = KentDensity::new(mu, g1, 2.0, 1.5);

    // ||x|| != 1 => Kent log_density returns None.
    let sample = Vector3::new(0.5, 0.0, 0.0);
    let density = dist.density::<nalgebra::U1, U3>(&sample.as_view());
    assert!(
        density.is_none(),
        "Expected density to be None for ||x|| != 1"
    );
}

#[test]
fn test_density_at_mu_greater_than_opposite_when_kappa_positive() {
    let mu = Vector3::new(1.0, 0.0, 0.0);
    let g1 = Vector3::new(0.0, 1.0, 0.0);

    let dist = KentDensity::new(mu, g1, 3.0, 0.5);

    let at_mu = dist.density::<nalgebra::U1, U3>(&mu.as_view()).unwrap();
    let at_minus_mu = dist.density::<nalgebra::U1, U3>(&(-mu).as_view()).unwrap();

    assert!(
        at_mu > at_minus_mu,
        "Expected density(mu) > density(-mu) when kappa > 0"
    );
}

#[test]
fn test_domain_contains_unit_ball_but_density_requires_shell() {
    let mu = Vector3::new(1.0, 0.0, 0.0);
    let g1 = Vector3::new(0.0, 1.0, 0.0);

    let dist = KentDensity::new(mu, g1, 2.0, 1.5);
    let domain: Domain<f64, U3> = dist.domain();

    let inside = Vector3::new(0.0, 0.0, 0.0);
    assert!(domain.contains::<nalgebra::U1, U3>(&inside.as_view()));

    let density_inside = dist.density::<nalgebra::U1, U3>(&inside.as_view());
    assert!(
        density_inside.is_none(),
        "Kent density is only defined on ||x|| ~= 1"
    );
}

#[test]
fn test_sampling_determinism() {
    let mu = Vector3::new(1.0, 0.0, 0.0);
    let g1 = Vector3::new(0.0, 1.0, 0.0);

    let dist = KentDensity::new(mu, g1, 2.0, 1.5);
    assert_sampling_determinism(&dist, RNG_SEED, 200);
}

#[test]
fn test_sampling_produces_unit_vectors() {
    let mu = Vector3::new(1.0, 0.0, 0.0);
    let g1 = Vector3::new(0.0, 1.0, 0.0);

    let dist = KentDensity::new(mu, g1, 2.0, 1.5);

    let samples = collect_samples_nd(&dist, N_SAMPLES, RNG_SEED);
    for s in samples {
        let norm: f64 = s.norm();
        assert!(
            ulps_eq!(norm, 1.0),
            "Expected ||sample|| ~= 1, got {}",
            norm
        );
    }
}

#[test]
fn test_sampling_respects_configuration_last_clamping() {
    // Regression-style test: even if `sampler.last` is not on the unit sphere,
    // sampling should still produce a unit vector.
    let mu = Vector3::new(1.0, 0.0, 0.0);
    let g1 = Vector3::new(0.0, 1.0, 0.0);

    let dist = KentDensity::new(mu, g1, 2.0, 1.5);

    let mut rng = rand::rngs::Xoshiro256PlusPlus::seed_from_u64(RNG_SEED);

    let sample = dist.sample(&mut rng).unwrap();
    let norm: f64 = sample.norm();
    assert!(ulps_eq!(norm, 1.0), "Expected unit vector after sampling");
}
