use crate::{Density, RejectionSampler, SamplingMode, domain::Domain, macros::tval};
use nalgebra::{Dim, OVector, RealField, SVector, U1, VectorView};
use rand::RngExt;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

/// A univariate normal PDF.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NormalDensity<T>(T, T, Domain<T, U1>)
where
    T: RealField;

impl<T> NormalDensity<T>
where
    T: RealField,
{
    /// Evaluates the cumulative distribution function at `x`.
    pub fn cdf(&self, x: T) -> T {
        let z = (x - self.0.clone()) / (self.1.clone() * tval!(2, usize).sqrt());

        tval!(0.5, f64) * (T::one() + Self::erf(z))
    }

    /// Evaluates the error function at `x`.
    pub fn erf(z: T) -> T {
        tval!(2, usize) / T::pi().sqrt()
            * (z.clone() - z.clone().powi(3) / tval!(3, usize)
                + z.clone().clone().powi(5) / tval!(10, usize)
                - z.clone().powi(7) / tval!(42, usize)
                + z.clone().powi(9) / tval!(216, usize)
                - z.clone().powi(11) / tval!(1320, usize))
    }

    /// Create a new [`NormalDensity`].
    pub fn new(mean: T, std_dev: T, opt_a: Option<T>, opt_b: Option<T>) -> Option<Self> {
        if std_dev <= T::zero() {
            return None;
        }

        let sdom = (opt_a.clone(), opt_b.clone());

        if opt_a.unwrap_or(T::neg(T::one())) >= opt_b.unwrap_or(T::one()) {
            return None;
        }

        let domain = Domain::new_mdomain(OVector::from_element_generic(U1, U1, sdom));

        Some(Self(mean, std_dev, domain))
    }

    /// Returns the maximum value of the domain.
    pub fn maximum(&self) -> Option<T> {
        match &self.2.inner().unwrap() {
            (_, Some(max)) => Some(max.clone()),
            _ => None,
        }
    }

    /// Returns the minimum value of the domain.
    pub fn minimum(&self) -> Option<T> {
        match &self.2.inner().unwrap() {
            (Some(min), _) => Some(min.clone()),
            _ => None,
        }
    }
}

impl<T> Density<T, U1> for &NormalDensity<T>
where
    T: RealField,
    StandardNormal: Distribution<T>,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, U1, RStride, CStride>,
    ) -> Option<T> {
        if !self.2.contains(sample) {
            return None;
        }

        Some(
            T::one() / (self.1.clone() * tval!(2.0 * std::f64::consts::PI, f64).sqrt())
                * (-((sample[0].clone() - self.0.clone()) / self.1.clone()).powi(2)
                    / tval!(2, usize))
                .exp(),
        )
    }

    fn domain(&self) -> Domain<T, U1> {
        self.2.clone()
    }

    fn sample(&self, rng: &mut impl RngExt, mode: &SamplingMode) -> Option<SVector<T, 1>> {
        self.rejection_sample(rng, mode)
    }

    fn sample_iter(&self, rng: &mut impl RngExt) -> impl Iterator<Item = Option<OVector<T, U1>>> {
        let normal = StandardNormal;

        rng.sample_iter(normal).map(move |z| {
            let candidate = self.1.clone() * z + self.0.clone();

            if self
                .2
                .contains::<U1, U1>(&SVector::from([candidate.clone()]).as_view())
            {
                Some(OVector::from([candidate]))
            } else {
                None
            }
        })
    }
}

impl<T> RejectionSampler<T, U1> for &NormalDensity<T>
where
    T: RealField,
    StandardNormal: Distribution<T>,
{
    fn generate_candidate(&self, rng: &mut impl RngExt) -> OVector<T, U1> {
        let z = rng.sample(StandardNormal);

        OVector::from([self.1.clone() * z + self.0.clone()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_normal_density_at_mean() {
        let normal = NormalDensity::new(0.1, 0.2, None, None).unwrap();

        assert!(ulps_eq!(normal.cdf(-0.1), 0.15865588083956078));
        assert!(ulps_eq!(normal.cdf(0.1), 0.5));
        assert!(ulps_eq!(NormalDensity::erf(0.71), 0.6846642286867719));
    }

    #[test]
    fn test_normal_sample_iter_unbounded() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let normal = NormalDensity::new(0.0, 1.0, None, None).unwrap();

        // Collect first 100 samples from the iterator
        let samples: Vec<_> = (&normal)
            .sample_iter(&mut rng)
            .take(100)
            .flatten()
            .collect();

        // All samples should be Some (unbounded, so no rejections)
        assert_eq!(samples.len(), 100);

        // Samples should have reasonable mean and variance
        let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        let variance: f64 =
            samples.iter().map(|s| (s[0] - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        assert!(mean.abs() < 0.3, "Mean should be close to 0, got {}", mean);
        assert!(
            (variance - 1.0).abs() < 0.3,
            "Variance should be close to 1, got {}",
            variance
        );
    }

    #[test]
    fn test_normal_sample_iter_bounded() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let normal = NormalDensity::new(0.0, 1.0, Some(-0.5), Some(0.5)).unwrap();

        // Collect samples from the iterator (some will be None due to rejection)
        let samples: Vec<_> = (&normal)
            .sample_iter(&mut rng)
            .take(500)
            .flatten()
            .take(50)
            .collect();

        // Should get at least some valid samples
        assert!(!samples.is_empty());

        // All samples should be within bounds
        for sample in &samples {
            assert!(
                sample[0] >= -0.5 && sample[0] <= 0.5,
                "Sample out of bounds: {}",
                sample[0]
            );
        }
    }

    #[test]
    fn test_normal_sample_iter_rejection_pattern() {
        // Tests that rejection sampling produces mixed Some/None results
        // when bounds are very tight relative to the standard deviation.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let normal = NormalDensity::new(0.0, 0.1, Some(-0.05), Some(0.05)).unwrap();

        // Very tight bounds [-0.05, 0.05], σ=0.1 → expect ~68% acceptance
        let results: Vec<_> = (&normal).sample_iter(&mut rng).take(100).collect();

        // Some should be None due to rejection
        let none_count = results.iter().filter(|r| r.is_none()).count();
        assert!(
            none_count > 0,
            "Expected some None values due to rejection sampling"
        );

        // Some should be Some
        let some_count = results.iter().filter(|r| r.is_some()).count();
        assert!(some_count > 0, "Expected some valid samples");
    }

    #[test]
    fn test_normal_sample_iter_custom_params() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        let mean = 2.5;
        let std_dev = 0.5;
        let normal = NormalDensity::new(mean, std_dev, None, None).unwrap();

        let samples: Vec<_> = (&normal)
            .sample_iter(&mut rng)
            .take(100)
            .flatten()
            .collect();

        assert_eq!(samples.len(), 100);

        let sample_mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        assert!(
            (sample_mean - mean).abs() < 0.3,
            "Sample mean should be close to {}, got {}",
            mean,
            sample_mean
        );
    }

    #[test]
    fn test_normal_very_small_std_dev() {
        // Test numerical stability with very small standard deviation
        // σ = 1e-10 should produce valid (though extreme) density values
        let normal = NormalDensity::new(0.0, 1e-10, None, None).unwrap();

        let dens_at_mean: f64 = (&normal)
            .density::<U1, U1>(&SVector::from([0.0]).as_view())
            .unwrap();

        // Density should be very high but finite
        // Peak of Normal(0, 1e-10) is 1/(1e-10 * √(2π)) ≈ 3.99e9
        assert!(dens_at_mean.is_finite(), "Density should be finite");
        assert!(
            dens_at_mean > 1e8,
            "Density should be very high for small σ"
        );

        // Slightly off from mean should have lower density
        let dens_small_offset: f64 = (&normal)
            .density::<U1, U1>(&SVector::from([1e-11]).as_view())
            .unwrap();

        assert!(dens_small_offset.is_finite());
        assert!(dens_small_offset > 0.0);
        assert!(
            dens_small_offset < dens_at_mean,
            "Density should decrease away from mean"
        );
    }

    #[test]
    fn test_normal_very_large_std_dev() {
        // Test numerical stability with very large standard deviation
        // σ = 1e6 should produce valid (though small) density values
        let normal = NormalDensity::new(0.0, 1e6, None, None).unwrap();

        let dens_at_mean: f64 = (&normal)
            .density::<U1, U1>(&SVector::from([0.0]).as_view())
            .unwrap();

        // Density at mean is 1/(σ * √(2π))
        // For σ = 1e6: 1/(1e6 * √(2π)) ≈ 3.99e-7
        assert!(dens_at_mean.is_finite());
        assert!(dens_at_mean > 0.0);
        assert!(
            dens_at_mean < 1e-5,
            "Density should be very small for large σ"
        );

        // Far from mean should have reasonable small density
        let dens_far: f64 = (&normal)
            .density::<U1, U1>(&SVector::from([1e6]).as_view())
            .unwrap();

        assert!(dens_far.is_finite());
        assert!(dens_far > 0.0);
        assert!(dens_far < dens_at_mean);
    }

    #[test]
    fn test_normal_sampling_produces_valid_samples() {
        // Basic sampling test for consistency with other distributions
        let normal = NormalDensity::new(0.0, 1.0, None, None).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        for _ in 0..100 {
            let sample = (&normal)
                .sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
                .unwrap();

            // Sample should be valid
            assert_eq!(sample.len(), 1);
            let sample_val: f64 = sample[0];
            assert!(sample_val.is_finite());
        }
    }

    #[test]
    fn test_normal_rejects_inverted_bounds() {
        // Should reject when upper bound < lower bound
        let result = NormalDensity::new(0.0, 1.0, Some(1.0), Some(0.0));
        assert!(
            result.is_none(),
            "NormalDensity should reject inverted bounds (upper < lower)"
        );
    }

    #[test]
    fn test_normal_rejects_zero_std_dev() {
        // Should reject zero standard deviation - degenerate case
        let result = NormalDensity::new(0.0, 0.0, None, None);
        assert!(
            result.is_none(),
            "NormalDensity should reject zero standard deviation"
        );
    }

    #[test]
    fn test_normal_rejects_negative_std_dev() {
        // Should reject negative standard deviation
        let result = NormalDensity::new(0.0, -1.0, None, None);
        assert!(
            result.is_none(),
            "NormalDensity should reject negative standard deviation"
        );
    }

    #[test]
    fn test_normal_sampling_bounded_narrow_range_statistical_validation() {
        // Test with bounds that are tighter than the std_dev
        // Bounds width: 0.2, std_dev: 0.1, so width < std_dev
        let normal = &NormalDensity::new(0.0, 0.1, Some(-0.1), Some(0.1)).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let mut samples = Vec::new();
        for _ in 0..5000 {
            if let Some(s) =
                (&normal).sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
            {
                samples.push(s[0]);
            }
        }

        assert!(!samples.is_empty(), "Should successfully sample");

        // Verify mean is approximately correct
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(
            (mean - 0.0).abs() < 0.02,
            "Sample mean should be ≈0.0, got {}",
            mean
        );

        // Verify std_dev is smaller than the truncated space [width = 0.2]
        // For uniform distribution in [-0.1, 0.1], std_dev = 0.2/sqrt(12) ≈ 0.0577
        // For truncated normal, it will be smaller than this but still bounded by 0.1
        let variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let sample_std_dev = variance.sqrt();

        // Bounds span [-0.1, 0.1], width = 0.2
        // Sample std_dev should be less than half the bounds width (0.1)
        assert!(
            sample_std_dev < 0.1,
            "Sample std_dev ({}) should be less than bounds width",
            sample_std_dev
        );
    }

    #[test]
    fn test_normal_statistical_validation_standard() {
        // Standard normal: μ=0.0, σ=1.0
        let normal = &NormalDensity::new(0.0, 1.0, None, None).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        let samples: Vec<f64> = (0..10000)
            .filter_map(|_| {
                normal
                    .sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
                    .map(|s| s[0])
            })
            .collect();

        assert!(!samples.is_empty(), "Should generate samples");

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        assert!(
            (mean - 0.0).abs() < 0.1,
            "Mean should be ≈0.0, got {}",
            mean
        );
        assert!(
            (variance - 1.0).abs() < 0.3,
            "Variance should be ≈1.0, got {}",
            variance
        );
    }

    #[test]
    fn test_normal_statistical_validation_shifted_scaled() {
        // Shifted/scaled: μ=5.0, σ=2.0
        let normal = &NormalDensity::new(5.0, 2.0, None, None).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(43);

        let samples: Vec<f64> = (0..10000)
            .filter_map(|_| {
                normal
                    .sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
                    .map(|s| s[0])
            })
            .collect();

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        assert!(
            (mean - 5.0).abs() < 0.1,
            "Mean should be ≈5.0, got {}",
            mean
        );
        assert!(
            (variance - 4.0).abs() < 0.3,
            "Variance should be ≈4.0, got {}",
            variance
        );
    }

    #[test]
    fn test_normal_statistical_validation_negative_small_std() {
        // Negative mean, small std_dev: μ=-10.0, σ=0.5
        let normal = &NormalDensity::new(-10.0, 0.5, None, None).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(44);

        let samples: Vec<f64> = (0..10000)
            .filter_map(|_| {
                normal
                    .sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
                    .map(|s| s[0])
            })
            .collect();

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        assert!(
            (mean - (-10.0)).abs() < 0.1,
            "Mean should be ≈-10.0, got {}",
            mean
        );
        assert!(
            (variance - 0.25).abs() < 0.3,
            "Variance should be ≈0.25, got {}",
            variance
        );
    }

    #[test]
    fn test_normal_statistical_validation_bounded_large_values() {
        // Large values with bounds: μ=100.0, σ=10.0, bounds=[80, 120]
        let normal = &NormalDensity::new(100.0, 10.0, Some(80.0), Some(120.0)).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(45);

        let samples: Vec<f64> = (0..10000)
            .filter_map(|_| {
                normal
                    .sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
                    .map(|s| s[0])
            })
            .collect();

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        // Verify all samples are within bounds
        assert!(
            samples.iter().all(|&s| (80.0..=120.0).contains(&s)),
            "All samples should be within [80, 120]"
        );

        assert!(
            (mean - 100.0).abs() < 0.1,
            "Mean should be ≈100.0, got {}",
            mean
        );
        // Note: variance is reduced by truncation to bounds [80, 120]
        // Unbounded variance would be 100.0, but bounds reduce it to ~77-80
        assert!(
            (variance - 100.0).abs() < 30.0,
            "Variance should be close to 100.0 (allowing for truncation), got {}",
            variance
        );
    }

    #[test]
    fn test_normal_statistical_validation_tight_bounds() {
        // Tight bounds: μ=0.0, σ=0.1, bounds=[-0.05, 0.05]
        let normal = &NormalDensity::new(0.0, 0.1, Some(-0.05), Some(0.05)).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(46);

        let samples: Vec<f64> = (0..10000)
            .filter_map(|_| {
                normal
                    .sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
                    .map(|s| s[0])
            })
            .collect();

        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        // Verify all samples are within bounds
        assert!(
            samples.iter().all(|&s| (-0.05..=0.05).contains(&s)),
            "All samples should be within [-0.05, 0.05]"
        );

        assert!(
            (mean - 0.0).abs() < 0.1,
            "Mean should be ≈0.0, got {}",
            mean
        );
        // Note: variance is reduced by truncation, so we just check it's reasonable
        assert!(
            variance < 0.01,
            "Variance should be small due to tight bounds, got {}",
            variance
        );
    }
}
