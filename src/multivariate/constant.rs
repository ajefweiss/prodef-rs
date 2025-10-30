use std::iter::repeat;

use crate::{Density, SamplingMode, domain::Domain};
use nalgebra::{Dim, OVector, RealField, SVector, U1, VectorView};
use rand::RngExt;
use serde::{Deserialize, Serialize};

/// A constant (point mass) PDF.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ConstantDensity<T>(Domain<T, U1>)
where
    T: RealField;

impl<T> ConstantDensity<T>
where
    T: RealField,
{
    /// Create a new [`ConstantDensity`].
    pub fn new(constant: T) -> Self {
        Self(Domain::new_mdomain(OVector::from_element_generic(
            U1,
            U1,
            (Some(constant.clone()), Some(constant)),
        )))
    }

    /// Returns the constant value.
    pub fn constant(&self) -> T {
        match &self.0.inner().unwrap() {
            (Some(constant), Some(_)) => constant.clone(),
            // Safe: ConstantDensity constructor creates MDomain with (c, c) bounds,
            // ensuring both min and max are always Some(c).
            _ => unreachable!("ConstantDensity MDomain always has explicit equal bounds"),
        }
    }
}

impl<T> Density<T, U1> for &ConstantDensity<T>
where
    T: RealField,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, U1, RStride, CStride>,
    ) -> Option<T> {
        if !self.0.contains(sample) {
            return None;
        }

        Some(T::one())
    }

    fn domain(&self) -> Domain<T, U1> {
        self.0.clone()
    }

    fn sample(&self, _rng: &mut impl RngExt, _mode: &SamplingMode) -> Option<SVector<T, 1>> {
        match &self.0.inner().unwrap() {
            (Some(constant), Some(_)) => Some(SVector::from([constant.clone()])),
            // Safe by construction
            _ => unreachable!(),
        }
    }

    fn sample_iter(&self, _rng: &mut impl RngExt) -> impl Iterator<Item = Option<OVector<T, U1>>> {
        match &self.0.inner().unwrap() {
            (Some(constant), Some(_)) => repeat(Some(OVector::from([constant.clone()]))),
            // Safe by construction
            _ => unreachable!(),
        }
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
    fn test_constant_density() {
        let constant = ConstantDensity::new(5.0);
        assert!(ulps_eq!(constant.constant(), 5.0));
    }

    #[test]
    fn test_constant_sampling_produces_valid_samples() {
        let constant = ConstantDensity::new(2.5);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        for _ in 0..50 {
            let sample = (&constant)
                .sample(&mut rng, &SamplingMode::SingleAttempt)
                .unwrap();
            assert!(ulps_eq!(sample[0], 2.5));
        }
    }

    #[test]
    fn test_constant_outside_domain() {
        let constant = &ConstantDensity::new(0.0);
        assert!(
            constant
                .density::<U1, U1>(&OVector::from([1.0]).as_view())
                .is_none()
        );
    }

    #[test]
    fn test_constant_sample_iter_returns_same_value() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let constant = ConstantDensity::new(3.7);

        let samples: Vec<_> = (&constant)
            .sample_iter(&mut rng)
            .take(100)
            .flatten()
            .collect();

        assert_eq!(samples.len(), 100);

        // All samples should be exactly the constant value
        for sample in &samples {
            assert!(ulps_eq!(sample[0], 3.7));
        }
    }

    #[test]
    fn test_constant_sample_iter_different_values() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);

        for constant_val in &[1.0, 2.5, -5.0, 0.0, 100.0] {
            let constant = ConstantDensity::new(*constant_val);

            let samples: Vec<_> = (&constant)
                .sample_iter(&mut rng)
                .take(50)
                .flatten()
                .collect();

            assert_eq!(samples.len(), 50);

            for sample in &samples {
                assert!(ulps_eq!(sample[0], *constant_val));
            }
        }
    }

    #[test]
    fn test_constant_sample_iter_always_valid() {
        // Constant distribution always returns the same value deterministically.
        // No randomness or rejection, so all results are always Some.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let constant = ConstantDensity::new(1.5);

        let results: Vec<_> = (&constant).sample_iter(&mut rng).take(100).collect();

        let all_valid = results.iter().all(|r| r.is_some());
        assert!(
            all_valid,
            "Constant deterministic sampling always produces valid results"
        );

        // All should be the exact constant value
        for result in results.iter().flatten() {
            assert_eq!(result[0], 1.5);
        }
    }
}
