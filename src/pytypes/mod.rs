//! Python types to be used with `pyo3`.
//!
//! The types in this module are designed to be used with the `pyo3` library, which allows for seamless interoperability between Rust and Python.

mod multinormal;
mod multivariate;
mod particle;
mod sampling;
mod univariate;

pub use multinormal::*;
pub use multivariate::*;
pub use particle::*;
pub use sampling::*;
pub use univariate::*;

#[cfg(feature = "pyo3_f32")]
type Float = f32;
#[cfg(not(feature = "pyo3_f32"))]
type Float = f64;
