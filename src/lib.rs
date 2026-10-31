#![doc = include_str!("../README.md")]

mod density;
mod domain;
mod kent;
mod multinormal;
mod multivariate;
mod particle;
mod sampling;
mod univariate;

// Enable access to Python types, only if the "pyo3" feature is enabled.
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

/// A helper macro that converts a value of a known primitive type to a generic (by default `T`).
macro_rules! generic_value {
    ($expr:expr, bool) => {
        T::from_f32($expr).expect("failed to convert the expresion (bool) to a generic type")
    };
    ($expr:expr, bool, $generic:expr) => {
        T::from_f32($expr).expect("failed to convert the expresion (bool) to a generic type")
    };
    ($expr:expr, f32) => {
        T::from_f32($expr).expect("failed to convert the expresion (f32) to a generic type")
    };
    ($expr:expr, f32, $generic:expr) => {
        T::from_f32($expr).expect("failed to convert the expresion (f32) to a generic type")
    };
    ($expr:expr, f64) => {
        T::from_f64($expr).expect("failed to convert the expresion (f64) to a generic type")
    };
    ($expr:expr, f64, $generic:expr) => {
        T::from_f64($expr).expect("failed to convert the expresion (f64) to a generic type")
    };
    ($expr:expr, isize) => {
        T::from_usize($expr).expect("failed to convert the expresion (isize) to a generic type")
    };
    ($expr:expr, isize, $generic:expr) => {
        $generic::from_usize($expr)
            .expect("failed to convert the expresion (isize) to a generic type")
    };
    ($expr:expr, usize) => {
        T::from_usize($expr).expect("failed to convert the expresion (usize) to a generic type")
    };
    ($expr:expr, usize, $generic:expr) => {
        $generic::from_usize($expr)
            .expect("failed to convert the expresion (usize) to a generic type")
    };
}

pub(crate) use generic_value;
