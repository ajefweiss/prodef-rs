use crate::{Density, SamplingMode, domain::Domain, macros::tval};
use nalgebra::{Dim, OVector, RealField, SVector, Scalar, U1, VectorView};
use rand::RngExt;
use rand_distr::{Uniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};

/// A loguniform PDF.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LogUniformDensity<T>(Domain<T, U1>)
where
    T: Scalar;

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

impl<T> Density<T, U1> for LogUniformDensity<T>
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

    fn mean(&self) -> SVector<T, 1> {
        let a = self.minimum();
        let b = self.maximum();
        SVector::from([(b.clone() - a.clone()) / (b.ln() - a.ln())])
    }

    fn sample(&self, rng: &mut impl RngExt, _mode: &SamplingMode) -> Option<SVector<T, 1>> {
        let uniform = Uniform::new_inclusive(
            self.0.minimum_values()[0].clone().unwrap().ln(),
            self.0.maximum_values()[0].clone().unwrap().ln(),
        )
        .unwrap();

        Some(SVector::from([rng.sample(uniform).exp()]))
    }

    fn sample_iter(&self, rng: &mut impl RngExt) -> impl Iterator<Item = Option<SVector<T, 1>>> {
        let uniform = Uniform::new_inclusive(
            self.0.minimum_values()[0].clone().unwrap().ln(),
            self.0.maximum_values()[0].clone().unwrap().ln(),
        )
        .unwrap();

        rng.sample_iter(uniform)
            .map(|value| Some(OVector::from_element_generic(U1, U1, value.exp())))
    }

    fn variance(&self) -> SVector<T, 1> {
        // For a log-uniform distribution on [a, b], the variance is:
        // variance = E[X²] - (E[X])²
        // where E[X²] = (b² - a²) / (2 * (ln(b) - ln(a)))
        let a = self.minimum();
        let b = self.maximum();
        let ln_ratio = b.clone().ln() - a.clone().ln();

        let a_sq = a.clone() * a;
        let b_sq = b.clone() * b;
        let e_x_sq = (b_sq - a_sq) / (tval!(2.0, f64) * ln_ratio);

        let mean = self.mean()[0].clone();
        let mean_sq = mean.clone() * mean;

        SVector::from([e_x_sq - mean_sq])
    }
}

impl<T> TryFrom<crate::univariate::UnivariateDensity<T>> for LogUniformDensity<T>
where
    T: RealField,
{
    type Error = ();

    fn try_from(value: crate::univariate::UnivariateDensity<T>) -> Result<Self, Self::Error> {
        match value {
            crate::univariate::UnivariateDensity::Loguniform(pdf) => Ok(pdf),
            _ => Err(()),
        }
    }
}
