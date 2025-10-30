use crate::{Density, SamplingMode, domain::Domain};
use nalgebra::{Dim, OVector, RealField, SVector, U1, VectorView};
use rand::RngExt;
use rand_distr::{Uniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};

/// A loguniform PDF.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LogUniformDensity<T>(Domain<T, U1>)
where
    T: RealField;

impl<T> LogUniformDensity<T>
where
    T: RealField,
{
    /// Create a new [`LogUniformDensity`].
    ///
    /// Returns `None` if:
    /// - `a >= b` (inverted or degenerate bounds)
    /// - `a <= 0` or `b <= 0` (log domain requires positive values)
    pub fn new(a: T, b: T) -> Option<Self> {
        if a >= b || a <= T::zero() || b <= T::zero() {
            None
        } else {
            Some(Self(Domain::new_mdomain(OVector::from_element_generic(
                U1,
                U1,
                (Some(a), Some(b)),
            ))))
        }
    }

    /// Returns the maximum value of the domain.
    pub fn maximum(&self) -> T {
        match &self.0.inner().unwrap() {
            (_, Some(max)) => max.clone(),
            // Safe: LogUniformDensity constructor validates a < b condition
            // and creates MDomain with explicit bounds, guaranteeing (min, max) both Some.
            _ => unreachable!("MDomain always has explicit bounds in LogUniformDensity"),
        }
    }

    /// Returns the minimum value of the domain.
    pub fn minimum(&self) -> T {
        match &self.0.inner().unwrap() {
            (Some(min), _) => min.clone(),
            // Safe: LogUniformDensity constructor validates a < b condition
            // and creates MDomain with explicit bounds, guaranteeing (min, max) both Some.
            _ => unreachable!("MDomain always has explicit bounds in LogUniformDensity"),
        }
    }
}

impl<T> Density<T, U1> for &LogUniformDensity<T>
where
    T: RealField + SampleUniform,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, U1, RStride, CStride>,
    ) -> Option<T> {
        if !self.0.contains(sample) {
            return None;
        }

        Some(
            T::one()
                / (sample[0].clone()
                    * (self.0.maximum_values()[0].clone().unwrap().ln()
                        - self.0.minimum_values()[0].clone().unwrap().ln())),
        )
    }

    fn domain(&self) -> Domain<T, U1> {
        self.0.clone()
    }

    fn sample(&self, rng: &mut impl RngExt, _mode: &SamplingMode) -> Option<SVector<T, 1>> {
        let uniform = Uniform::new_inclusive(
            self.0.minimum_values()[0].clone().unwrap().ln(),
            self.0.maximum_values()[0].clone().unwrap().ln(),
        )
        .unwrap();

        Some(SVector::from([rng.sample(uniform).exp()]))
    }

    fn sample_iter(&self, rng: &mut impl RngExt) -> impl Iterator<Item = Option<OVector<T, U1>>> {
        let uniform = Uniform::new_inclusive(
            self.0.minimum_values()[0].clone().unwrap().ln(),
            self.0.maximum_values()[0].clone().unwrap().ln(),
        )
        .unwrap();

        rng.sample_iter(uniform)
            .map(|value| Some(OVector::from_element_generic(U1, U1, value.exp())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;
    use nalgebra::{SVector, U1};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_loguniform_density() {
        let loguniform = LogUniformDensity::new(1.0, 10.0).unwrap();
        let sample = SVector::from([1.0]);

        // density = 1 / (x * ln(b/a)) = 1 / (1.0 * ln(10)) = 1 / 2.302585...
        let expected = 1.0 / (1.0 * (10.0_f64.ln() - 1.0_f64.ln()));
        assert!(ulps_eq!(
            (&loguniform).density::<U1, U1>(&sample.as_view()).unwrap(),
            expected,
            epsilon = 1e-10
        ));
    }

    #[test]
    fn test_loguniform_invalid_bounds() {
        // Inverted bounds (a > b)
        assert!(LogUniformDensity::new(10.0, 1.0).is_none());

        // Equal bounds (a == b) is now rejected as degenerate
        assert!(LogUniformDensity::new(1.0, 1.0).is_none());
    }

    #[test]
    fn test_loguniform_rejects_degenerate_range() {
        // Should reject range (a, a) - degenerate case
        let a = 1.0;
        let result = LogUniformDensity::new(a, a);
        assert!(
            result.is_none(),
            "LogUniformDensity should reject degenerate range (a, a)"
        );
    }

    #[test]
    fn test_loguniform_outside_domain() {
        let loguniform = LogUniformDensity::new(1.0, 10.0).unwrap();

        // Sample outside domain
        let sample_below = SVector::from([0.5]);
        let sample_above = SVector::from([20.0]);

        assert!(
            (&loguniform)
                .density::<U1, U1>(&sample_below.as_view())
                .is_none()
        );
        assert!(
            (&loguniform)
                .density::<U1, U1>(&sample_above.as_view())
                .is_none()
        );
    }

    #[test]
    fn test_loguniform_sampling_produces_valid_samples() {
        let loguniform = LogUniformDensity::new(1.0, 10.0).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        for _ in 0..100 {
            let sample = (&loguniform)
                .sample(&mut rng, &SamplingMode::SingleAttempt)
                .unwrap();
            let view = sample.as_view();

            // Check sample is within domain
            assert!(loguniform.0.contains::<U1, U1>(&view));

            // Check sample returns valid density
            assert!((&loguniform).density::<U1, U1>(&view).is_some());

            // Check sample is in expected range
            assert!(sample[0] >= 1.0 && sample[0] <= 10.0);
        }
    }

    #[test]
    fn test_loguniform_log_uniformity() {
        // Test that the distribution is uniform in log space
        let loguniform = LogUniformDensity::new(1.0, 10.0).unwrap();

        let sample_1 = SVector::from([1.0]);
        let sample_10 = SVector::from([10.0]);
        let sample_3 = SVector::from([3.0]);

        let density_1: f64 = (&loguniform)
            .density::<U1, U1>(&sample_1.as_view())
            .unwrap();
        let density_10: f64 = (&loguniform)
            .density::<U1, U1>(&sample_10.as_view())
            .unwrap();
        let density_3: f64 = (&loguniform)
            .density::<U1, U1>(&sample_3.as_view())
            .unwrap();

        // All densities should be positive and finite
        assert!(density_1.is_finite() && density_1 > 0.0);
        assert!(density_10.is_finite() && density_10 > 0.0);
        assert!(density_3.is_finite() && density_3 > 0.0);

        // Density should decrease with x (since 1/(x * ln(b/a)))
        assert!(density_1 > density_3);
        assert!(density_3 > density_10);
    }

    #[test]
    fn test_loguniform_sample_iter_log_space_validity() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let loguniform = LogUniformDensity::new(1.0, 10.0).unwrap();

        let samples: Vec<_> = (&loguniform)
            .sample_iter(&mut rng)
            .take(100)
            .flatten()
            .collect();

        assert_eq!(samples.len(), 100);

        // All samples should be within bounds
        for sample in &samples {
            assert!(
                sample[0] >= 1.0 && sample[0] <= 10.0,
                "Sample out of bounds: {}",
                sample[0]
            );
        }
    }

    #[test]
    fn test_loguniform_sample_iter_log_space_statistical_coverage() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        let loguniform = LogUniformDensity::new(1.0, 100.0).unwrap();

        let samples: Vec<OVector<f64, U1>> = (&loguniform)
            .sample_iter(&mut rng)
            .take(500)
            .flatten()
            .collect();

        assert!(samples.len() > 100);

        // Convert to log space and check uniformity
        let log_samples: Vec<f64> = samples.iter().map(|s| s[0].ln()).collect();
        let log_min = log_samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let log_max = log_samples
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // In log space, should span the range [ln(1), ln(100)]
        let expected_min = 1.0_f64.ln();
        let expected_max = 100.0_f64.ln();
        assert!((log_min - expected_min).abs() < 0.5);
        assert!((log_max - expected_max).abs() < 0.5);
    }

    #[test]
    fn test_loguniform_sample_iter_always_valid() {
        // LogUniform uses inverse-transform sampling:
        // exp(U(ln(a), ln(b))) always produces samples within [a, b].
        // No rejection sampling, so all results are always Some.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let loguniform = LogUniformDensity::new(1.0, 10.0).unwrap();

        let results: Vec<_> = (&loguniform).sample_iter(&mut rng).take(100).collect();

        let all_valid = results.iter().all(|r| r.is_some());
        assert!(
            all_valid,
            "LogUniform inverse-transform sampling always produces valid results"
        );

        // Verify samples are within bounds
        for result in results.iter().flatten() {
            assert!(result[0] >= 1.0 && result[0] <= 10.0);
        }
    }

    #[test]
    fn test_loguniform_tight_bounds() {
        // Test numerical stability with very tight bounds in log space
        // Bounds [1, 1.001] have ln ratio ≈ 0.0009995
        let loguniform = LogUniformDensity::new(1.0, 1.001).unwrap();

        let dens_lower: f64 = (&loguniform)
            .density::<U1, U1>(&SVector::from([1.0]).as_view())
            .unwrap();

        let dens_upper: f64 = (&loguniform)
            .density::<U1, U1>(&SVector::from([1.001]).as_view())
            .unwrap();

        let dens_mid: f64 = (&loguniform)
            .density::<U1, U1>(&SVector::from([1.0005]).as_view())
            .unwrap();

        // All should be finite and positive
        assert!(dens_lower.is_finite() && dens_lower > 0.0);
        assert!(dens_upper.is_finite() && dens_upper > 0.0);
        assert!(dens_mid.is_finite() && dens_mid > 0.0);

        // Density should decrease as x increases (1/(x * ln(b/a)))
        assert!(dens_lower > dens_mid);
        assert!(dens_mid > dens_upper);
    }

    #[test]
    fn test_loguniform_rejects_inverted_bounds() {
        // Should reject when upper bound < lower bound
        let result = LogUniformDensity::new(10.0, 1.0);
        assert!(
            result.is_none(),
            "LogUniformDensity should reject inverted bounds (upper < lower)"
        );
    }
}
