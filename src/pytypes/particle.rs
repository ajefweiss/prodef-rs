use crate::{multinormal::MultivariateNormalDensity, particle::ParticleDensity, pytypes::Float};
use nalgebra::Dyn;
use numpy::PyReadonlyArray2;
use pyo3::{PyResult, exceptions::PyTypeError, prelude::*};

/// A particle (non-parametric) density for use in Python.
#[derive(Clone)]
#[pyclass(from_py_object, name = "ParticleDensity")]
pub struct PyParticleDensity {
    #[pyo3(get)]
    names: Vec<String>,
    particles: Vec<Vec<Float>>,
    weights: Option<Vec<Float>>,
}

impl PyParticleDensity {
    /// Return a reference to the underlying particles.
    pub fn particles(&self) -> &Vec<Vec<Float>> {
        &self.particles
    }

    /// Return a reference to the particle weights.
    pub fn weights(&self) -> &Option<Vec<Float>> {
        &self.weights
    }
}

impl From<ParticleDensity<Float, Dyn, MultivariateNormalDensity<Float, Dyn>>>
    for PyParticleDensity
{
    /// Convert a [`ParticleDensity`] to a [`PyParticleDensity`].
    fn from(density: ParticleDensity<Float, Dyn, MultivariateNormalDensity<Float, Dyn>>) -> Self {
        let particles_matrix = density.particles();
        let n_particles = particles_matrix.ncols();
        let n_dims = particles_matrix.nrows();

        let particles = (0..n_particles)
            .map(|col| particles_matrix.column(col).iter().cloned().collect())
            .collect();

        let weights = density
            .weights()
            .as_ref()
            .map(|w| w.iter().cloned().collect());

        let names = (0..n_dims).map(|i| format!("dim_{}", i)).collect();

        Self {
            names,
            particles,
            weights,
        }
    }
}

#[pymethods]
impl PyParticleDensity {
    /// Create a new [`PyParticleDensity`] from a 2D numpy array of particles.
    #[new]
    pub fn new(particles: PyReadonlyArray2<Float>, weights: Option<Vec<Float>>) -> PyResult<Self> {
        let matrix = match particles.try_as_matrix::<Dyn, Dyn, Dyn, Dyn>() {
            Some(value) => Ok(value.transpose()),
            None => Err(PyTypeError::new_err(
                "conversion of a numpy array to nalgebra matrix failed",
            )),
        }?;

        let n_dims = matrix.nrows();
        let particles = (0..matrix.ncols())
            .map(|col| matrix.column(col).iter().cloned().collect())
            .collect();

        let names = (0..n_dims).map(|i| format!("dim_{}", i)).collect();

        Ok(Self {
            names,
            particles,
            weights,
        })
    }

    /// Return the number of particles.
    pub fn count(&self) -> usize {
        self.particles.len()
    }

    /// Return the dimension of each particle.
    pub fn dimension(&self) -> usize {
        self.particles.first().map(|p| p.len()).unwrap_or(0)
    }

    /// Return whether particles are weighted.
    pub fn is_weighted(&self) -> bool {
        self.weights.is_some()
    }

    /// Return the names of the dimensions.
    pub fn names(&self) -> Vec<String> {
        self.names.clone()
    }
}
