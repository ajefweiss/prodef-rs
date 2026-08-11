use crate::{
    pytypes::Float,
    univariate::{
        ConstantDensity, CosineDensity, LogUniformDensity, NormalDensity, UniformDensity,
        UnivariateDensity,
    },
};
use pyo3::{PyResult, exceptions::PyValueError, prelude::*, types::PyType};

/// A univariate density for use in Python.
#[derive(Clone)]
#[pyclass(from_py_object, name = "Univariate")]
pub struct PyUnivariate {
    #[pyo3(get)]
    name: String,
    density: UnivariateDensity<Float>,
}

impl PyUnivariate {
    /// Return a reference to the underlying [`UnivariateDensity`].
    pub fn inner(&self) -> &UnivariateDensity<Float> {
        &self.density
    }

    /// Return the parameter name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Create a new [`PyUnivariate`] with a given name and density.
    pub fn new(name: String, density: UnivariateDensity<Float>) -> Self {
        Self { name, density }
    }
}

impl From<UnivariateDensity<Float>> for PyUnivariate {
    /// Convert a [`UnivariateDensity`] to a [`PyUnivariate`] with an auto-generated name.
    fn from(density: UnivariateDensity<Float>) -> Self {
        let name = match &density {
            UnivariateDensity::Constant(_) => "constant".to_string(),
            UnivariateDensity::Cosine(_) => "cosine".to_string(),
            UnivariateDensity::Lognormal(_) => "lognormal".to_string(),
            UnivariateDensity::Loguniform(_) => "loguniform".to_string(),
            UnivariateDensity::Normal(_) => "normal".to_string(),
            UnivariateDensity::Uniform(_) => "uniform".to_string(),
        };
        Self { name, density }
    }
}

#[pymethods]
impl PyUnivariate {
    /// Create a new [`PyUnivariate`] with a constant density.
    #[classmethod]
    pub fn constant(_cls: &Bound<PyType>, name: String, value: Float) -> PyResult<Self> {
        Ok(Self {
            name,
            density: ConstantDensity::new(value).into(),
        })
    }

    /// Create a new [`PyUnivariate`] with a cosine density.
    #[classmethod]
    pub fn cosine(
        _cls: &Bound<PyType>,
        name: String,
        minimum: Float,
        maximum: Float,
    ) -> PyResult<Self> {
        match CosineDensity::new(minimum, maximum) {
            Some(value) => Ok(Self {
                name,
                density: value.into(),
            }),
            None => Err(PyValueError::new_err("invalid domain")),
        }
    }

    /// Return the domain of the univariate density.
    pub fn domain(&self) -> (Option<Float>, Option<Float>) {
        match &self.density {
            UnivariateDensity::Constant(constant) => {
                (Some(constant.constant()), Some(constant.constant()))
            }
            UnivariateDensity::Cosine(cosine) => (Some(cosine.minimum()), Some(cosine.maximum())),
            UnivariateDensity::Lognormal(log_normal) => {
                (Some(log_normal.minimum()), Some(log_normal.maximum()))
            }
            UnivariateDensity::Loguniform(log_uniform) => {
                (Some(log_uniform.minimum()), Some(log_uniform.maximum()))
            }
            UnivariateDensity::Normal(normal) => (normal.minimum(), normal.maximum()),
            UnivariateDensity::Uniform(uniform) => {
                (Some(uniform.minimum()), Some(uniform.maximum()))
            }
        }
    }

    /// Create a new [`PyUnivariate`] with a normal density.
    #[classmethod]
    #[pyo3(signature = (name, mean, std_dev, opt_minimum = None, opt_maximum = None))]
    pub fn normal(
        _cls: &Bound<PyType>,
        name: String,
        mean: Float,
        std_dev: Float,
        opt_minimum: Option<Float>,
        opt_maximum: Option<Float>,
    ) -> PyResult<Self> {
        match NormalDensity::new(mean, std_dev, opt_minimum, opt_maximum) {
            Some(value) => Ok(Self {
                name,
                density: value.into(),
            }),
            None => Err(PyValueError::new_err("invalid domain")),
        }
    }

    /// Create a new [`PyUnivariate`] with a log-uniform density.
    #[classmethod]
    pub fn loguniform(
        _cls: &Bound<PyType>,
        name: String,
        minimum: Float,
        maximum: Float,
    ) -> PyResult<Self> {
        match LogUniformDensity::new(minimum, maximum) {
            Some(value) => Ok(Self {
                name,
                density: value.into(),
            }),
            None => Err(PyValueError::new_err("invalid domain")),
        }
    }

    /// Create a new [`PyUnivariate`] with a uniform density.
    #[classmethod]
    pub fn uniform(
        _cls: &Bound<PyType>,
        name: String,
        minimum: Float,
        maximum: Float,
    ) -> PyResult<Self> {
        match UniformDensity::new(minimum, maximum) {
            Some(value) => Ok(Self {
                name,
                density: value.into(),
            }),
            None => Err(PyValueError::new_err("invalid domain")),
        }
    }

    /// Return the type name of the univariate density.
    pub fn typename(&self) -> &str {
        match &self.density {
            UnivariateDensity::Constant(_) => "constant",
            UnivariateDensity::Cosine(_) => "cosine",
            UnivariateDensity::Lognormal(_) => "lognormal",
            UnivariateDensity::Loguniform(_) => "loguniform",
            UnivariateDensity::Normal(_) => "normal",
            UnivariateDensity::Uniform(_) => "uniform",
        }
    }
}
