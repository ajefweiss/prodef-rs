use crate::{KentDensity, pytypes::Float};
use nalgebra::{U1, U3};
use numpy::PyReadonlyArray1;
use pyo3::{PyResult, exceptions::PyTypeError, prelude::*};

/// A multinormal density for use in Python.
#[derive(Clone)]
#[pyclass(from_py_object, name = "KentDensity")]
pub struct PyKentDensity(pub KentDensity<Float>);

#[pymethods]
impl PyKentDensity {
    /// Create a new [`PyKentDensity`].
    #[new]
    #[pyo3(signature = (mu, g1, kappa, beta))]
    pub fn new<'py>(
        mu: PyReadonlyArray1<Float>,
        g1: PyReadonlyArray1<Float>,
        kappa: Float,
        beta: Float,
    ) -> PyResult<Self> {
        let mu = match mu.try_as_matrix::<U1, U3, U1, U1>() {
            Some(value) => Ok(value.transpose()),
            None => Err(PyTypeError::new_err(
                "conversion of a numpy array to nalgebra matrix failed",
            )),
        }?;

        let g1 = match g1.try_as_matrix::<U1, U3, U1, U1>() {
            Some(value) => Ok(value.transpose()),
            None => Err(PyTypeError::new_err(
                "conversion of a numpy array to nalgebra matrix failed",
            )),
        }?;

        Ok(Self(KentDensity::new(mu, g1, kappa, beta)))
    }
}
