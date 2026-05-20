use crate::{Domain, multinormal::MultivariateNormalDensity, pytypes::Float};
use nalgebra::{DVector, Dyn, U1};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{Bound, PyResult, exceptions::PyTypeError, prelude::*, types::PyList};

/// A multinormal density for use in Python.
#[derive(Clone)]
#[pyclass(from_py_object, name = "MultiNormal")]
pub struct PyMultivariateNormal {
    mvnpdf: MultivariateNormalDensity<Float, Dyn>,
    #[pyo3(get)]
    names: Vec<String>,
}

impl From<MultivariateNormalDensity<Float, Dyn>> for PyMultivariateNormal {
    /// Convert a [`MultivariateNormalDensity`] to a [`PyMultivariateNormal`].
    fn from(mvnpdf: MultivariateNormalDensity<Float, Dyn>) -> Self {
        let dim = mvnpdf.covariance_matrix().nrows();
        let names = (0..dim).map(|i| format!("dim_{}", i)).collect();
        Self { names, mvnpdf }
    }
}

impl From<(MultivariateNormalDensity<Float, Dyn>, Vec<String>)> for PyMultivariateNormal {
    /// Convert a [`MultivariateNormalDensity`] to a [`PyMultivariateNormal`] with custom dimension names.
    fn from(tuple: (MultivariateNormalDensity<Float, Dyn>, Vec<String>)) -> Self {
        let (mvnpdf, names) = tuple;
        Self { names, mvnpdf }
    }
}

impl AsRef<MultivariateNormalDensity<Float, Dyn>> for PyMultivariateNormal {
    /// Get a reference to the underlying [`MultivariateNormalDensity`].
    fn as_ref(&self) -> &MultivariateNormalDensity<Float, Dyn> {
        &self.mvnpdf
    }
}

#[pymethods]
impl PyMultivariateNormal {
    /// Create a new [`PyMultivariateNormal`] with an unbounded domain.
    #[new]
    #[pyo3(signature = (mean, covariance, domain=None))]
    pub fn new<'py>(
        mean: PyReadonlyArray1<Float>,
        covariance: PyReadonlyArray2<Float>,
        domain: Option<Bound<'py, PyList>>,
    ) -> PyResult<Self> {
        let mean = match mean.try_as_matrix::<U1, Dyn, U1, Dyn>() {
            Some(value) => Ok(value.transpose()),
            None => Err(PyTypeError::new_err(
                "conversion of a numpy array to nalgebra matrix failed",
            )),
        }?;

        let matrix = match covariance.try_as_matrix::<Dyn, Dyn, Dyn, Dyn>() {
            Some(value) => Ok(value.transpose()),
            None => Err(PyTypeError::new_err(
                "conversion of a numpy array to nalgebra matrix failed",
            )),
        }?;

        let domain = match domain {
            Some(sdoms) => {
                let sdoms = sdoms
                    .iter()
                    .map(|d| d.extract::<(Option<Float>, Option<Float>)>().ok())
                    .collect::<Vec<Option<(Option<Float>, Option<Float>)>>>();

                if sdoms
                    .iter()
                    .any(|min_max| min_max.is_none() || (min_max.unwrap().1 < min_max.unwrap().0))
                {
                    return Err(PyTypeError::new_err(
                        "failed to convert one of the members of the domain argument to a PySDomain type",
                    ));
                }

                Domain::new_mdomain(DVector::from(
                    sdoms
                        .into_iter()
                        .map(|value| value.unwrap())
                        .collect::<Vec<(Option<Float>, Option<Float>)>>(),
                ))
            }
            None => Domain::new_udomain(Dyn(matrix.nrows())),
        };

        let result = MultivariateNormalDensity::new(matrix.clone(), domain, Some(mean));

        match result {
            Some(mvnpdf) => {
                let dim = matrix.nrows();
                let names = (0..dim).map(|i| format!("dim_{}", i)).collect();
                Ok(Self { names, mvnpdf })
            }
            None => Err(PyTypeError::new_err(
                "failed to construct multinormal density, dimensions of the arguments must be consistent",
            )),
        }
    }

    /// Return the names of the dimensions.
    pub fn names(&self) -> Vec<String> {
        self.names.clone()
    }
}
