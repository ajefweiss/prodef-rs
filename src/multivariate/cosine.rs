use crate::{Density, SamplingMode, domain::Domain};
use nalgebra::{Dim, OVector, RealField, SVector, U1, VectorView};
use rand::RngExt;
use rand_distr::{Uniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};

/// A cosine PDF defined on [-π/2, π/2].
///
/// Sine degenerate ranges (where sin is equal at endpoints) are rejected.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CosineDensity<T>(Domain<T, U1>)
where
    T: RealField;

impl<T> CosineDensity<T>
where
    T: RealField,
{
    /// Returns the maximum value of the domain.
    pub fn maximum(&self) -> T {
        match &self.0.inner().unwrap() {
            (_, Some(max)) => max.clone(),
            // Safe: CosineDensity constructor enforces MDomain with explicit bounds
            // after validating sine uniqueness and range constraints.
            _ => unreachable!("CosineDensity MDomain always has explicit bounds"),
        }
    }

    /// Returns the minimum value of the domain.
    pub fn minimum(&self) -> T {
        match &self.0.inner().unwrap() {
            (Some(min), _) => min.clone(),
            // Safe: CosineDensity constructor enforces MDomain with explicit bounds
            // after validating sine uniqueness and range constraints.
            _ => unreachable!("CosineDensity MDomain always has explicit bounds"),
        }
    }

    /// Create a new [`CosineDensity`].
    ///
    /// Returns [`None`] for an invalid domain range.
    pub fn new(minimum: T, maximum: T) -> Option<Self> {
        match (
            minimum > -T::frac_pi_2(),
            maximum < T::frac_pi_2(),
            maximum > minimum,
            maximum.clone().sin() != minimum.clone().sin(),
        ) {
            (true, true, true, true) => Some(Self(Domain::new_mdomain(
                OVector::from_element_generic(U1, U1, (Some(minimum), Some(maximum))),
            ))),
            _ => None,
        }
    }
}

impl<T> Density<T, U1> for &CosineDensity<T>
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

        Some(sample[0].clone().cos())
    }

    fn domain(&self) -> Domain<T, U1> {
        self.0.clone()
    }

    fn sample(&self, rng: &mut impl RngExt, _mode: &SamplingMode) -> Option<SVector<T, 1>> {
        // The range is limited to the interval [-π/2, π/2].
        // This invariant is guaranteed by the constructor.
        match &self.0.inner().unwrap() {
            (Some(min), Some(max)) => {
                let uniform = Uniform::new_inclusive(min.clone().sin(), max.clone().sin()).unwrap();

                Some(SVector::from([rng.sample(uniform).asin()]))
            }
            // Safe by construction
            _ => unreachable!(),
        }
    }

    fn sample_iter(&self, rng: &mut impl RngExt) -> impl Iterator<Item = Option<OVector<T, U1>>> {
        // The range is limited to the interval [-π/2, π/2].
        // This invariant is guaranteed by the constructor.
        match &self.0.inner().unwrap() {
            (Some(min), Some(max)) => {
                let uniform = Uniform::new_inclusive(min.clone().sin(), max.clone().sin()).unwrap();

                rng.sample_iter(uniform)
                    .map(|value| Some(OVector::from_element_generic(U1, U1, value.asin())))
            }
            // Safe by construction
            _ => unreachable!(),
        }
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
    fn test_cosine_density_value() {
        let cosine = CosineDensity::new(-0.5, 0.5).unwrap();
        let sample = SVector::from([0.0]);

        // At 0, cos(0) = 1
        assert!(ulps_eq!(
            (&cosine).density::<U1, U1>(&sample.as_view()).unwrap(),
            1.0,
            epsilon = 1e-10
        ));
    }

    #[test]
    fn test_cosine_invalid_bounds() {
        // Lower bound below -π/2
        assert!(CosineDensity::new(-2.0, 0.0).is_none());

        // Upper bound above π/2
        assert!(CosineDensity::new(0.0, 2.0).is_none());

        // Inverted bounds
        assert!(CosineDensity::new(0.5, -0.5).is_none());
    }

    #[test]
    fn test_cosine_outside_domain() {
        let cosine = CosineDensity::new(-0.5, 0.5).unwrap();

        // Sample outside domain
        let sample_below = SVector::from([-1.0]);
        let sample_above = SVector::from([1.0]);

        assert!(
            (&cosine)
                .density::<U1, U1>(&sample_below.as_view())
                .is_none()
        );
        assert!(
            (&cosine)
                .density::<U1, U1>(&sample_above.as_view())
                .is_none()
        );
    }

    #[test]
    fn test_cosine_sampling_produces_valid_samples() {
        let cosine = CosineDensity::new(-0.3, 0.3).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        for _ in 0..100 {
            let sample = (&cosine)
                .sample(&mut rng, &SamplingMode::SingleAttempt)
                .unwrap();
            let view = sample.as_view();

            // Check sample is within domain
            assert!(cosine.0.contains::<U1, U1>(&view));

            // Check sample returns valid density
            assert!((&cosine).density::<U1, U1>(&view).is_some());
        }
    }

    #[test]
    fn test_cosine_symmetry() {
        let cosine = CosineDensity::new(-0.3, 0.3).unwrap();

        let pos_sample = SVector::from([0.2]);
        let neg_sample = SVector::from([-0.2]);

        // cos is even, so density should be symmetric
        assert!(ulps_eq!(
            (&cosine).density::<U1, U1>(&pos_sample.as_view()).unwrap(),
            (&cosine).density::<U1, U1>(&neg_sample.as_view()).unwrap(),
            epsilon = 1e-10
        ));
    }

    #[test]
    fn test_cosine_sample_iter_bounded_validity() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let cosine = CosineDensity::new(-0.5, 0.5).unwrap();

        let samples: Vec<_> = (&cosine)
            .sample_iter(&mut rng)
            .take(100)
            .flatten()
            .collect();

        assert_eq!(samples.len(), 100);

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
    fn test_cosine_sample_iter_statistical_coverage() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        let cosine = CosineDensity::new(
            -std::f64::consts::PI / 2.0 + 0.01,
            std::f64::consts::PI / 2.0 - 0.01,
        )
        .unwrap();

        let samples: Vec<_> = (&cosine)
            .sample_iter(&mut rng)
            .take(200)
            .flatten()
            .collect();

        // Should get samples across the domain
        assert!(samples.len() > 50);

        // Check that samples are spread across the domain
        let min = samples.iter().map(|s| s[0]).fold(f64::INFINITY, f64::min);
        let max = samples
            .iter()
            .map(|s| s[0])
            .fold(f64::NEG_INFINITY, f64::max);

        // Range should cover a significant portion of the domain
        assert!((max - min) > 2.0, "Samples should be spread across domain");
    }

    #[test]
    fn test_cosine_sample_iter_always_valid() {
        // Cosine uses inverse-transform sampling via arcsin:
        // sin⁻¹(U(sin(a), sin(b))) always produces samples within [a, b].
        // No rejection sampling is performed, so all results are always Some.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let cosine = CosineDensity::new(-0.3, 0.3).unwrap();

        let results: Vec<_> = (&cosine).sample_iter(&mut rng).take(100).collect();

        let all_valid = results.iter().all(|r| r.is_some());
        assert!(
            all_valid,
            "Cosine inverse-transform sampling always produces valid results"
        );

        // Verify samples are within bounds
        for result in results.iter().flatten() {
            assert!(result[0] >= -0.3 && result[0] <= 0.3);
        }
    }

    #[test]
    fn test_cosine_rejects_degenerate_range() {
        // Should reject range (a, a) - degenerate case
        let a = 0.5;
        let result = CosineDensity::new(a, a);
        assert!(
            result.is_none(),
            "CosineDensity should reject degenerate range (a, a)"
        );
    }
}
