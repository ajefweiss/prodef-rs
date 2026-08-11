#![doc = include_str!("../README.md")]

mod density;
mod domain;
mod kent;
mod multinormal;
mod multivariate;
mod particle;
mod sampling;
mod univariate;

// Enable access to the Python types if the "pyo3" feature is enabled.
#[cfg(feature = "pyo3")]
pub mod pytypes;

#[cfg(test)]
mod tests;

pub use density::Density;
pub use domain::Domain;
pub use kent::KentDensity;
pub use multinormal::MultivariateNormalDensity;
pub use multivariate::MultivariateDensity;
pub use particle::ParticleDensity;
pub use univariate::{
    ConstantDensity, CosineDensity, LogUniformDensity, LognormalDensity, NormalDensity,
    UniformDensity, UnivariateDensity,
};

/// Converts a value to `T`.
macro_rules! tval {
    ($expr:expr, usize) => {
        T::from_usize($expr).unwrap()
    };
    ($expr:expr, f64) => {
        T::from_f64($expr).unwrap()
    };
}

pub(crate) use tval;
