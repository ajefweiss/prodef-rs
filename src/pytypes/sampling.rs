use crate::{
    Density, SamplingMode,
    pytypes::{Float, PyMultivariate, PyMultivariateNormal, PyUnivariate},
};
use numpy::{PyArray2, ToPyArray};
use pyo3::{PyResult, prelude::*, types::PyType};
use rand_xoshiro::{Xoshiro256PlusPlus, rand_core::SeedableRng};

/// A random number generator for use in Python.
#[derive(Clone)]
#[pyclass(from_py_object, name = "SamplingRng")]
pub struct PySamplingRng(Xoshiro256PlusPlus);

#[pymethods]
impl PySamplingRng {
    /// Create a new [`PySamplingRng`] with a random seed.
    #[classmethod]
    #[pyo3(signature = (seed = None))]
    pub fn new(_cls: &Bound<PyType>, seed: Option<u64>) -> PyResult<Self> {
        let rng = match seed {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::seed_from_u64(42),
        };

        Ok(Self(rng))
    }
}

/// Sampling strategy for the [`Density::sample`] function.
#[derive(Clone)]
#[pyclass(from_py_object, name = "SamplingMode")]
pub struct PySamplingMode(SamplingMode);

#[pymethods]
impl PySamplingMode {
    /// Sample once, and return the sample if it is valid, or [`None`] if it is not.
    #[classmethod]
    pub fn single_attempt(_cls: &Bound<PyType>) -> PyResult<Self> {
        Ok(Self(SamplingMode::SingleAttempt))
    }

    /// Sample until a valid sample is found, or the maximum number of attempts is reached.
    #[classmethod]
    pub fn until_valid(_cls: &Bound<PyType>, max_attempts: usize) -> PyResult<Self> {
        Ok(Self(SamplingMode::UntilValid { max_attempts }))
    }

    /// Sample until a valid sample is found, or the maximum number of attempts is reached, but clamp the sample to the domain if the threshold is exceeded.
    #[classmethod]
    pub fn until_valid_or_clamp(_cls: &Bound<PyType>, max_attempts: usize) -> PyResult<Self> {
        Ok(Self(SamplingMode::UntilValidOrClamp { max_attempts }))
    }

    /// Sample until a valid sample is found, with no maximum number of attempts.
    #[classmethod]
    pub fn until_valid_no_limit(_cls: &Bound<PyType>) -> PyResult<Self> {
        Ok(Self(SamplingMode::UntilValidNoLimit))
    }

    /// Return the type name of the sampling mode.
    pub fn typename(&self) -> &str {
        match &self.0 {
            SamplingMode::SingleAttempt => "single_attempt",
            SamplingMode::UntilValid { .. } => "until_valid",
            SamplingMode::UntilValidOrClamp { .. } => "until_valid_or_clamp",
            SamplingMode::UntilValidNoLimit => "until_valid_no_limit",
        }
    }
}

#[pymethods]
impl PyUnivariate {
    /// Sample a value from the distribution, returning [`None`] if the sample is invalid.
    pub fn sample(&self, rng: &mut PySamplingRng, mode: &PySamplingMode) -> Option<Float> {
        self.density()
            .sample(&mut rng.0, &mode.0)
            .map(|sample| sample[0])
    }
}

#[pymethods]
impl PyMultivariate {
    /// Sample a value from the distribution, returning [`None`] if the sample is invalid.
    pub fn sample(&self, rng: &mut PySamplingRng, mode: &PySamplingMode) -> Option<Float> {
        self.uvpdfs().iter().try_fold(1.0 as Float, |acc, uvpdf| {
            uvpdf.sample(rng, mode).map(|sample| acc * sample)
        })
    }
}

#[pymethods]
impl PyMultivariateNormal {
    /// Sample a value from the distribution, returning [`None`] if the sample is invalid.
    pub fn sample<'py>(
        &self,
        py: Python<'py>,
        rng: &mut PySamplingRng,
        mode: &PySamplingMode,
    ) -> Option<Bound<'py, PyArray2<Float>>> {
        self.as_multinormal_density()
            .sample(&mut rng.0, &mode.0)
            .map(|sample| sample.to_pyarray(py))
    }
}
