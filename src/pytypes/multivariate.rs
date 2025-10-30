use crate::{
    multivariate::MultivariateDensity,
    pytypes::{Float, PyUnivariate},
};
use nalgebra::{DefaultAllocator, Dim, OVector, U1, allocator::Allocator};
use pyo3::{PyResult, exceptions::PyTypeError, prelude::*, types::PyList};

/// A multivariate density for use in Python.
#[derive(Clone)]
#[pyclass(from_py_object, name = "Multivariate")]
pub struct PyMultivariate {
    #[pyo3(get)]
    uvpdfs: Vec<PyUnivariate>,
}

impl PyMultivariate {
    /// Create a [`MultivariateDensity`] from self.
    pub fn as_multivariate_density<N: Dim>(&self) -> MultivariateDensity<Float, N>
    where
        DefaultAllocator: Allocator<N>,
    {
        MultivariateDensity::new(OVector::from_iterator_generic(
            N::from_usize(self.uvpdfs.len()),
            U1,
            self.uvpdfs.iter().map(|u| u.density().clone()),
        ))
    }

    /// Return a reference to the underlying univariate densities.
    pub fn uvpdfs(&self) -> &Vec<PyUnivariate> {
        &self.uvpdfs
    }
}

#[pymethods]
impl PyMultivariate {
    #[new]
    /// Create a new [`PyMultivariate`] from a list of [`PyUnivariate`]s.
    pub fn new<'py>(priors: Bound<'py, PyList>) -> PyResult<Self> {
        let priors = priors
            .iter()
            .map(|p| match p.extract::<PyUnivariate>() {
                Ok(uvpdf) => Some(uvpdf.clone()),
                Err(_) => None,
            })
            .collect::<Vec<Option<PyUnivariate>>>();

        if priors.iter().any(|r| r.is_none()) {
            Err(PyTypeError::new_err(
                "failed to convert one of the members of the list argument to a PyUnivariate type",
            ))
        } else {
            Ok(Self {
                uvpdfs: priors.into_iter().map(|uvpdf| uvpdf.unwrap()).collect(),
            })
        }
    }

    /// Return the names of the underlying univariate densities.
    pub fn names(&self) -> Vec<String> {
        self.uvpdfs.iter().map(|u| u.name().to_string()).collect()
    }

    /// Return the type names of the underlying univariate densities.
    pub fn typenames(&self) -> Vec<String> {
        self.uvpdfs
            .iter()
            .map(|u| u.typename().to_string())
            .collect()
    }
}
