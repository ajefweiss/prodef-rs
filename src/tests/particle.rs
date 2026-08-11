use crate::{Density, Domain, ParticleDensity};
use approx::assert_abs_diff_eq;
use nalgebra::{Dyn, SVector, U1, U2};
use rand::{SeedableRng, rngs::Xoshiro256PlusPlus};

use super::*;

#[test]
fn test_particle_constructor_valid() {
    use nalgebra::OMatrix;

    let mut particles = OMatrix::<f64, U2, Dyn>::zeros(3);
    particles[(0, 0)] = 0.0;
    particles[(1, 0)] = 0.0;
    particles[(0, 1)] = 1.0;
    particles[(1, 1)] = 1.0;
    particles[(0, 2)] = -1.0;
    particles[(1, 2)] = -1.0;

    let domain = Domain::new_udomain(U2);
    let _dist = ParticleDensity::from_vectors::<Dyn, Dyn>(&particles.as_view(), domain, None, None)
        .unwrap();
}

#[test]
fn test_particle_unweighted_sampling() {
    use nalgebra::OMatrix;

    let mut particles = OMatrix::<f64, U2, Dyn>::zeros(3);
    particles[(0, 0)] = 0.0;
    particles[(1, 0)] = 0.0;
    particles[(0, 1)] = 1.0;
    particles[(1, 1)] = 1.0;
    particles[(0, 2)] = -1.0;
    particles[(1, 2)] = -1.0;

    let domain = Domain::new_udomain(U2);
    let dist = ParticleDensity::from_vectors::<Dyn, Dyn>(&particles.as_view(), domain, None, None)
        .unwrap();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(RNG_SEED);
    let samples: Vec<_> = (0..100).filter_map(|_| dist.sample(&mut rng)).collect();

    assert_eq!(
        samples.len(),
        100,
        "Should generate correct number of samples"
    );
}

#[test]
fn test_particle_sampling_determinism() {
    use nalgebra::OMatrix;

    let mut particles = OMatrix::<f64, U2, Dyn>::zeros(3);
    particles[(0, 0)] = 0.0;
    particles[(1, 0)] = 0.0;
    particles[(0, 1)] = 1.0;
    particles[(1, 1)] = 1.0;
    particles[(0, 2)] = -1.0;
    particles[(1, 2)] = -1.0;

    let domain = Domain::new_udomain(U2);
    let dist = ParticleDensity::from_vectors::<Dyn, Dyn>(&particles.as_view(), domain, None, None)
        .unwrap();

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
fn test_particle_density_evaluation() {
    use nalgebra::OMatrix;

    let mut particles = OMatrix::<f64, U2, Dyn>::zeros(2);
    particles[(0, 0)] = 0.0;
    particles[(1, 0)] = 0.0;
    particles[(0, 1)] = 1.0;
    particles[(1, 1)] = 1.0;

    let domain = Domain::new_udomain(U2);
    let dist = ParticleDensity::from_vectors::<Dyn, Dyn>(&particles.as_view(), domain, None, None)
        .unwrap();

    // Evaluate at particle location
    let sample_at_particle = SVector::from([0.0, 0.0]);
    let density_at = dist.density::<U1, U2>(&sample_at_particle.as_view());

    // Should have non-zero density at particle location
    assert!(
        density_at.is_some(),
        "Density should exist at particle location"
    );
    assert!(
        density_at.unwrap() > 0.0,
        "Density at particle location should be positive"
    );
}

#[test]
fn test_particle_sampling_mean_convergence() {
    use nalgebra::OMatrix;

    // Create particles centered around (0, 0) and (2, 2)
    let mut particles = OMatrix::<f64, U2, Dyn>::zeros(4);
    particles[(0, 0)] = 0.0;
    particles[(1, 0)] = 0.0;
    particles[(0, 1)] = 0.0;
    particles[(1, 1)] = 0.0;
    particles[(0, 2)] = 2.0;
    particles[(1, 2)] = 2.0;
    particles[(0, 3)] = 2.0;
    particles[(1, 3)] = 2.0;

    let domain = Domain::new_udomain(U2);
    let dist = ParticleDensity::from_vectors::<Dyn, Dyn>(&particles.as_view(), domain, None, None)
        .unwrap();

    let samples = collect_samples_nd(&dist, N_SAMPLES, RNG_SEED);

    // Extract per-dimension samples
    let dim0_samples: Vec<f64> = samples.iter().map(|s| s[0]).collect();
    let dim1_samples: Vec<f64> = samples.iter().map(|s| s[1]).collect();

    let empirical_mean0 = compute_sample_mean(&dim0_samples);
    let empirical_mean1 = compute_sample_mean(&dim1_samples);

    // Empirical mean should be close to mean of particles: (0+0+2+2)/4 = 1.0
    assert_abs_diff_eq!(empirical_mean0, 1.0, epsilon = 0.1);
    assert_abs_diff_eq!(empirical_mean1, 1.0, epsilon = 0.1);
}

#[test]
fn test_particle_sampling_variance_convergence() {
    use nalgebra::OMatrix;

    // Create particles at corners of unit square
    let mut particles = OMatrix::<f64, U2, Dyn>::zeros(4);
    particles[(0, 0)] = 0.0;
    particles[(1, 0)] = 0.0;
    particles[(0, 1)] = 1.0;
    particles[(1, 1)] = 0.0;
    particles[(0, 2)] = 0.0;
    particles[(1, 2)] = 1.0;
    particles[(0, 3)] = 1.0;
    particles[(1, 3)] = 1.0;

    let domain = Domain::new_udomain(U2);
    let dist = ParticleDensity::from_vectors::<Dyn, Dyn>(&particles.as_view(), domain, None, None)
        .unwrap();

    let samples = collect_samples_nd(&dist, N_SAMPLES, RNG_SEED);

    // Extract per-dimension samples
    let dim0_samples: Vec<f64> = samples.iter().map(|s| s[0]).collect();
    let dim1_samples: Vec<f64> = samples.iter().map(|s| s[1]).collect();

    let dim0_mean = compute_sample_mean(&dim0_samples);
    let dim1_mean = compute_sample_mean(&dim1_samples);

    let empirical_var0 = compute_sample_variance(&dim0_samples, dim0_mean);
    let empirical_var1 = compute_sample_variance(&dim1_samples, dim1_mean);

    // Particles have variance, but kernel smoothing adds uncertainty
    // Just check that variances are reasonable and positive
    assert!(empirical_var0 > 0.0, "Variance should be positive");
    assert!(empirical_var1 > 0.0, "Variance should be positive");
    assert!(
        empirical_var0 < 1.0,
        "Variance should be bounded by particle spread"
    );
    assert!(
        empirical_var1 < 1.0,
        "Variance should be bounded by particle spread"
    );
}
