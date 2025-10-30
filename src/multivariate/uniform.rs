use crate::{Density, SamplingMode, domain::Domain};
use nalgebra::{Dim, OVector, RealField, SVector, U1, VectorView};
use rand::RngExt;
use rand_distr::{Uniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};

/// A uniform PDF.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct UniformDensity<T>(Domain<T, U1>)
where
    T: RealField;

impl<T> UniformDensity<T>
where
    T: RealField,
{
    /// Create a new [`UniformDensity`].
    pub fn new(a: T, b: T) -> Option<Self> {
        if a >= b {
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
            // Safe: UniformDensity constructor creates MDomain with explicit bounds (a, b)
            // where a < b, so upper bound is always Some(max).
            _ => unreachable!("MDomain always has explicit bounds in UniformDensity"),
        }
    }

    /// Returns the minimum value of the domain.
    pub fn minimum(&self) -> T {
        match &self.0.inner().unwrap() {
            (Some(min), _) => min.clone(),
            // Safe: UniformDensity constructor creates MDomain with explicit bounds (a, b)
            // where a < b, so lower bound is always Some(min).
            _ => unreachable!("MDomain always has explicit bounds in UniformDensity"),
        }
    }
}

impl<T> Density<T, U1> for &UniformDensity<T>
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
                / (self.0.maximum_values()[0].clone().unwrap()
                    - self.0.minimum_values()[0].clone().unwrap()),
        )
    }

    fn domain(&self) -> Domain<T, U1> {
        self.0.clone()
    }

    fn sample(&self, rng: &mut impl RngExt, _mode: &SamplingMode) -> Option<SVector<T, 1>> {
        let uniform = Uniform::new_inclusive(
            self.0.minimum_values()[0].clone().unwrap(),
            self.0.maximum_values()[0].clone().unwrap(),
        )
        .unwrap();

        Some(SVector::from([rng.sample(uniform)]))
    }

    fn sample_iter(&self, rng: &mut impl RngExt) -> impl Iterator<Item = Option<OVector<T, U1>>> {
        let uniform = Uniform::new_inclusive(
            self.0.minimum_values()[0].clone().unwrap(),
            self.0.maximum_values()[0].clone().unwrap(),
        )
        .unwrap();

        rng.sample_iter(uniform)
            .map(|value| Some(OVector::from([value])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::ulps_eq;
    use nalgebra::OVector;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_uniform_density() {
        let uniform = &UniformDensity::new(0.0, 1.0).unwrap();
        let sample = OVector::from([0.5]);
        assert!(ulps_eq!(
            uniform.density::<U1, U1>(&sample.as_view()).unwrap(),
            1.0,
            epsilon = 1e-10
        ));
    }

    #[test]
    fn test_uniform_outside_domain() {
        let uniform = &UniformDensity::new(0.0, 1.0).unwrap();
        assert!(
            uniform
                .density::<U1, U1>(&OVector::from([1.5]).as_view())
                .is_none()
        );
    }

    #[test]
    fn test_uniform_invalid_bounds() {
        assert!(UniformDensity::new(1.0, 0.0).is_none());
    }

    #[test]
    fn test_uniform_rejects_degenerate_range() {
        // Should reject range (a, a) - degenerate case
        let a = 0.5;
        let result = UniformDensity::new(a, a);
        assert!(
            result.is_none(),
            "UniformDensity should reject degenerate range (a, a)"
        );
    }

    #[test]
    fn test_uniform_sampling_produces_valid_samples() {
        let uniform = UniformDensity::new(0.0, 1.0).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        for _ in 0..100 {
            let sample = (&uniform)
                .sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
                .unwrap();
            assert!(sample[0] >= 0.0 && sample[0] <= 1.0);
        }
    }

    #[test]
    fn test_uniform_sample_iter_unbounded() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let uniform = UniformDensity::new(0.0, 1.0).unwrap();

        let samples: Vec<_> = (&uniform)
            .sample_iter(&mut rng)
            .take(100)
            .flatten()
            .collect();

        assert_eq!(samples.len(), 100);

        // All samples should be within [0, 1]
        for sample in &samples {
            assert!(sample[0] >= 0.0 && sample[0] <= 1.0);
        }

        // Samples should have mean close to 0.5
        let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        assert!(
            (mean - 0.5).abs() < 0.1,
            "Mean should be close to 0.5, got {}",
            mean
        );
    }

    #[test]
    fn test_uniform_sample_iter_custom_bounds() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        let a = 2.0;
        let b = 5.0;
        let uniform = UniformDensity::new(a, b).unwrap();

        let samples: Vec<_> = (&uniform)
            .sample_iter(&mut rng)
            .take(100)
            .flatten()
            .collect();

        assert_eq!(samples.len(), 100);

        // All samples should be within [a, b]
        for sample in &samples {
            assert!(
                sample[0] >= a && sample[0] <= b,
                "Sample out of bounds: {}",
                sample[0]
            );
        }

        // Mean should be close to (a+b)/2
        let expected_mean = (a + b) / 2.0;
        let mean: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        assert!(
            (mean - expected_mean).abs() < 0.2,
            "Mean should be close to {}, got {}",
            expected_mean,
            mean
        );
    }

    #[test]
    fn test_uniform_sample_iter_always_valid() {
        // Uniform distribution over [0, 1] always produces valid samples
        // since the distribution is defined entirely by the bounded domain.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let uniform = UniformDensity::new(0.0, 1.0).unwrap();

        let results: Vec<_> = (&uniform).sample_iter(&mut rng).take(100).collect();

        // All should be Some (by definition of Uniform bounds)
        let all_valid = results.iter().all(|r| r.is_some());
        assert!(all_valid, "Uniform sampling always produces valid results");

        // Verify all are in [0, 1]
        for result in results.iter().flatten() {
            assert!(result[0] >= 0.0 && result[0] <= 1.0);
        }
    }

    #[test]
    fn test_uniform_very_narrow_range() {
        // Test numerical stability with very narrow range
        // Range = 1e-10 should produce high but valid density
        let uniform = UniformDensity::new(0.0, 1e-10).unwrap();

        let dens: f64 = (&uniform)
            .density::<U1, U1>(&SVector::from([5e-11]).as_view())
            .unwrap();

        // Density = 1 / range = 1e10
        assert!(dens.is_finite());
        assert!(dens > 0.0);
        assert!(dens > 1e9, "Density should be very high for narrow range");

        // Outside range should be None
        let dens_outside = (&uniform).density::<U1, U1>(&SVector::from([2e-10]).as_view());

        assert!(dens_outside.is_none());
    }

    #[test]
    fn test_uniform_rejects_inverted_bounds() {
        // Should reject when upper bound < lower bound
        let result = UniformDensity::new(1.0, 0.5);
        assert!(
            result.is_none(),
            "UniformDensity should reject inverted bounds (upper < lower)"
        );
    }
}
