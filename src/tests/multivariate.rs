use crate::{Density, MultivariateDensity, NormalDensity, UniformDensity};
use approx::assert_abs_diff_eq;
use nalgebra::{SVector, U1, U2};

use super::*;

#[test]
fn test_multivariate_constructor_valid() {
    let univariates = SVector::from([
        UniformDensity::new(0.0, 1.0).unwrap().into(),
        NormalDensity::new(0.0, 1.0, None, None).unwrap().into(),
    ]);
    let _dist = MultivariateDensity::new(univariates);
}

#[test]
fn test_multivariate_sampling_containment() {
    let univariates = SVector::from([
        UniformDensity::new(0.0, 1.0).unwrap().into(),
        UniformDensity::new(-1.0, 1.0).unwrap().into(),
    ]);
    let dist = MultivariateDensity::new(univariates);
    let samples = collect_samples_nd(&dist, N_SAMPLES, RNG_SEED);

    for sample in &samples {
        assert!(sample[0] >= 0.0 && sample[0] <= 1.0, "Dim 0 out of bounds");
        assert!(sample[1] >= -1.0 && sample[1] <= 1.0, "Dim 1 out of bounds");
    }
}

#[test]
fn test_multivariate_sampling_determinism() {
    let univariates = SVector::from([
        UniformDensity::new(0.0, 1.0).unwrap().into(),
        UniformDensity::new(0.0, 1.0).unwrap().into(),
    ]);
    let dist = MultivariateDensity::new(univariates);

    let samples1 = collect_samples_nd(&dist, 50, RNG_SEED);
    let samples2 = collect_samples_nd(&dist, 50, RNG_SEED);

    assert_eq!(
        samples1, samples2,
        "Sampling not deterministic with same seed"
    );
}

#[test]
fn test_multivariate_density_evaluation() {
    let univariates = SVector::from([
        UniformDensity::new(0.0, 1.0).unwrap().into(),
        UniformDensity::new(0.0, 1.0).unwrap().into(),
    ]);
    let dist = MultivariateDensity::new(univariates);

    let sample_in = SVector::from([0.5, 0.5]);
    let sample_out = SVector::from([1.5, 0.5]);

    let density_in = dist.density::<U1, U2>(&sample_in.as_view());
    let density_out = dist.density::<U1, U2>(&sample_out.as_view());

    assert!(density_in.is_some(), "Density should exist inside domain");
    assert!(
        density_out.is_none(),
        "Density should not exist outside domain"
    );
}

#[test]
fn test_multivariate_sampling_mean_convergence() {
    let univariates = SVector::from([
        UniformDensity::new(0.0, 1.0).unwrap().into(),
        UniformDensity::new(1.0, 2.0).unwrap().into(),
    ]);
    let dist = MultivariateDensity::new(univariates);

    let samples = collect_samples_nd(&dist, N_SAMPLES, RNG_SEED);

    // Extract per-dimension samples
    let dim0_samples: Vec<f64> = samples.iter().map(|s| s[0]).collect();
    let dim1_samples: Vec<f64> = samples.iter().map(|s| s[1]).collect();

    let empirical_mean0 = compute_sample_mean(&dim0_samples);
    let empirical_mean1 = compute_sample_mean(&dim1_samples);

    // Theoretical means: Uniform(0,1) has mean 0.5, Uniform(1,2) has mean 1.5
    assert_abs_diff_eq!(empirical_mean0, 0.5, epsilon = 0.025);
    assert_abs_diff_eq!(empirical_mean1, 1.5, epsilon = 0.025);
}

#[test]
fn test_multivariate_sampling_variance_convergence() {
    let univariates = SVector::from([
        NormalDensity::new(0.0, 1.0, None, None).unwrap().into(),
        NormalDensity::new(0.0, 2.0, None, None).unwrap().into(),
    ]);
    let dist = MultivariateDensity::new(univariates);

    let samples = collect_samples_nd(&dist, N_SAMPLES, RNG_SEED);

    // Extract per-dimension samples
    let dim0_samples: Vec<f64> = samples.iter().map(|s| s[0]).collect();
    let dim1_samples: Vec<f64> = samples.iter().map(|s| s[1]).collect();

    let dim0_mean = compute_sample_mean(&dim0_samples);
    let dim1_mean = compute_sample_mean(&dim1_samples);

    let empirical_var0 = compute_sample_variance(&dim0_samples, dim0_mean);
    let empirical_var1 = compute_sample_variance(&dim1_samples, dim1_mean);

    // Theoretical variances: Normal(0,1) has variance 1.0, Normal(0,2) has variance 4.0 (sigma^2)
    assert_abs_diff_eq!(empirical_var0, 1.0, epsilon = 0.1);
    assert_abs_diff_eq!(empirical_var1, 4.0, epsilon = 0.25);
}
