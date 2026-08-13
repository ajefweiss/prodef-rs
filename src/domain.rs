//! Types and traits for representing function domains.

use approx::ulps_eq;
use nalgebra::{
    DefaultAllocator, Dim, OVector, RealField, Scalar, U1, VectorView, allocator::Allocator,
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// A generic function domain specifying valid input regions for PDFs.
///
/// Three domain types are currently supported:
///
/// - **Unbounded (`UDomain`)**: The entire ℝᵈ space (all `D`-dimensional real values valid)
/// - **Bounded (`MDomain`)**: A `D`-dimensional hypercube with per-dimension interval bounds
/// - **Spherical (`SDomain`)**: A `D`-dimensional ball (sphere) centered at the origin with radius 1
///
/// # Operations
///
/// **Domain checking**: Use `contains()` to validate if a sample is in the domain.
/// - Returns `true` for samples within bounds (inclusive), `false` otherwise
/// - Used by `Density::density()` to return `None` for out-of-domain samples
/// - For bounded domains, returns `true` if `a ≤ sample ≤ b` in all dimensions
/// - For spherical domains, returns `true` if `||sample|| ≤ 1`
///
/// **Boundary enforcement**: Use `clamp()` to project samples onto domain boundaries.
/// - Clamps each coordinate to its dimension's [min, max] range (inclusive)
/// - Used by `SamplingConfiguration::UntilValidOrClamp` after rejection sampling budget exhausted
/// - Result will always satisfy `contains()` unless domain is invalid
///
/// **Querying bounds**: Use `maximum_values()` and `minimum_values()` for per-dimension limits.
/// - Returns `Option<T>` per dimension (None if unbounded in that direction)
/// - `Some(x)` means the dimension is bounded at `x` (inclusive)
/// - `None` means the dimension is unbounded in that direction
///
/// # Examples
///
/// Create an unbounded domain (all of ℝ):
/// ```
/// # use prodef::Domain;
/// # use nalgebra::U1;
/// let domain = Domain::<f64, _>::new_udomain(U1);
/// // All values are valid in unbounded domain
/// ```
///
/// Create a bounded 1D domain (closed interval [0, 1]):
/// ```
/// # use prodef::Domain;
/// # use nalgebra::{OVector, U1};
/// let bounds = OVector::from([(Some(0.0), Some(1.0))]);
/// let domain = Domain::new_mdomain(bounds);
/// // Domain now restricts values to [0, 1] (INCLUSIVE on both ends)
/// // 0.0 and 1.0 are VALID samples
/// ```
///
/// Check containment in 2D (inclusive boundaries):
/// ```
/// # use prodef::Domain;
/// # use nalgebra::{SVector, OVector, U2, U1};
/// let bounds = OVector::from([(Some(0.0), Some(1.0)), (Some(-1.0), Some(1.0))]);
/// let domain = Domain::new_mdomain(bounds);
///
/// // Boundary points are INCLUDED
/// assert!(domain.contains::<U1, U2>(&SVector::from([0.0, -1.0]).as_view()));  // min corner
/// assert!(domain.contains::<U1, U2>(&SVector::from([1.0, 1.0]).as_view()));   // max corner
/// assert!(domain.contains::<U1, U2>(&SVector::from([0.5, 0.5]).as_view()));   // interior point
/// ```
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(bound(
    serialize = "D: Serialize, OVector<(Option<T>, Option<T>), D>: Serialize, OVector<T, D>: Serialize"
))]
#[serde(bound(
    deserialize = "D: Deserialize<'de>, OVector<(Option<T>, Option<T>), D>: Deserialize<'de>, OVector<T, D>: Deserialize<'de>"
))]
pub enum Domain<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    UDomain(D),
    MDomain(OVector<(Option<T>, Option<T>), D>),
    SDomain(D),
    ShDomain(D),
}

impl<T> Domain<T, U1>
where
    T: RealField,
{
    /// Returns the inner boundary values, if possible
    pub fn inner(&self) -> Option<(Option<T>, Option<T>)> {
        match self {
            Domain::UDomain(_) => None,
            Domain::MDomain(sdoms) => Some(sdoms[0].clone()),
            Domain::SDomain(_) => Some((Some(-T::one()), Some(T::one()))),
            Domain::ShDomain(_) => Some((Some(-T::one()), Some(T::one()))),
        }
    }
}

impl<T, D> Domain<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Clip `sample` to be contained within the domain.
    ///
    /// This function projects each coordinate onto the domain's valid range for that dimension.
    /// If the sample is already within the domain (as checked by `contains()`), it is returned unchanged.
    /// Clamped values are guaranteed to be within the closed interval [min, max] and will satisfy `contains()`.
    ///
    /// For unbounded dimensions, returns the sample value unchanged (no clamping applied).
    ///
    /// # Examples
    ///
    /// ```
    /// # use prodef::Domain;
    /// # use nalgebra::{SVector, OVector, U1};
    /// let bounds = OVector::from([(Some(0.0), Some(1.0))]);
    /// let domain = Domain::new_mdomain(bounds);
    ///
    /// // Values outside are clamped to boundaries
    /// let clamped_min = domain.clamp::<U1, U1>(&SVector::from([-0.5]).as_view());
    /// assert_eq!(clamped_min[0], 0.0);  // clamped to min boundary
    ///
    /// let clamped_max = domain.clamp::<U1, U1>(&SVector::from([1.5]).as_view());
    /// assert_eq!(clamped_max[0], 1.0);  // clamped to max boundary
    ///
    /// let unclamped = domain.clamp::<U1, U1>(&SVector::from([0.7]).as_view());
    /// assert_eq!(unclamped[0], 0.7);    // already inside, unchanged
    /// ```
    pub fn clamp<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, D, RStride, CStride>,
    ) -> OVector<T, D> {
        match self {
            Domain::UDomain(_) => sample.clone_owned(),
            Domain::MDomain(sdoms) => OVector::from_iterator_generic(
                sample.shape_generic().0,
                U1,
                sdoms.iter().enumerate().map(|(i, (opt_min, opt_max))| {
                    let value = &sample[i];

                    // Clamp value to the bounds, respecting unbounded dimensions (None)
                    if let Some(min) = opt_min
                        && value < min
                    {
                        return min.clone();
                    }

                    if let Some(max) = opt_max
                        && value > max
                    {
                        return max.clone();
                    }

                    value.clone()
                }),
            ),
            Domain::SDomain { .. } => {
                let norm = sample.norm();

                // If already inside the sphere, return unchanged
                if norm <= T::one() {
                    sample.clone_owned()
                } else {
                    // Project onto the sphere surface by scaling
                    sample.clone_owned() / norm
                }
            }
            Domain::ShDomain { .. } => {
                let norm = sample.norm();

                // Always project onto the sphere shell surface (radius = 1)
                if norm > T::zero() {
                    sample.clone_owned() / norm
                } else {
                    // Handle zero vector case: project to (1, 0, 0, ...)
                    let mut result = sample.clone_owned();
                    result[0] = T::one();
                    result
                }
            }
        }
    }

    /// Returns `true` if the sample is contained within the domain.
    ///
    /// # Examples
    ///
    /// ```
    /// # use prodef::Domain;
    /// # use nalgebra::{SVector, OVector, U1, U2};
    /// let bounds = OVector::from([(Some(0.0), Some(1.0))]);
    /// let domain = Domain::new_mdomain(bounds);
    ///
    /// // Boundary points are INCLUDED (inclusive semantics)
    /// assert!(domain.contains::<U1, U1>(&SVector::from([0.0]).as_view()));  // at min
    /// assert!(domain.contains::<U1, U1>(&SVector::from([1.0]).as_view()));  // at max
    /// assert!(domain.contains::<U1, U1>(&SVector::from([0.5]).as_view())); // interior
    /// assert!(!domain.contains::<U1, U1>(&SVector::from([-0.1]).as_view())); // outside
    /// ```
    pub fn contains<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, D, RStride, CStride>,
    ) -> bool {
        match self {
            Domain::UDomain(_) => true,
            Domain::MDomain(sdoms) => sdoms.iter().zip(sample).all(|(sdom, value)| match sdom {
                (Some(min), Some(max)) => (value >= min) & (value <= max),
                (Some(min), None) => value >= min,
                (None, Some(max)) => value <= max,
                (None, None) => true,
            }),
            Domain::SDomain { .. } => sample.norm() <= T::one(),
            Domain::ShDomain { .. } => {
                let norm = sample.norm();
                ulps_eq!(norm, T::one())
            }
        }
    }

    /// Returns the maximum value of the domain along each dimension.
    pub fn maximum_values(&self) -> OVector<Option<T>, D> {
        match self {
            Domain::UDomain(dim) => OVector::from_element_generic(*dim, U1, None),
            Domain::MDomain(sdoms) => OVector::from_iterator_generic(
                sdoms.shape_generic().0,
                U1,
                sdoms.iter().map(|sdom| sdom.1.clone()),
            ),
            Domain::SDomain(dim) => OVector::from_element_generic(*dim, U1, Some(T::one())),
            Domain::ShDomain(dim) => OVector::from_element_generic(*dim, U1, Some(T::one())),
        }
    }

    /// Returns the minimum value of the domain along each dimension.
    pub fn minimum_values(&self) -> OVector<Option<T>, D> {
        match self {
            Domain::UDomain(dim) => OVector::from_element_generic(*dim, U1, None),
            Domain::MDomain(sdoms) => OVector::from_iterator_generic(
                sdoms.shape_generic().0,
                U1,
                sdoms.iter().map(|sdom| sdom.0.clone()),
            ),
            Domain::SDomain(dim) => OVector::from_element_generic(*dim, U1, Some(-T::one())),
            Domain::ShDomain(dim) => OVector::from_element_generic(*dim, U1, Some(-T::one())),
        }
    }

    /// Create a new [`Domain`] from a vector of boundary values.
    pub fn new_mdomain(domains: OVector<(Option<T>, Option<T>), D>) -> Self {
        Domain::MDomain(domains)
    }

    /// Create a new unbounded domain.
    pub fn new_udomain(dim: D) -> Self {
        Domain::UDomain(dim)
    }

    /// Create a new spherical domain centered at the origin with radius 1.
    pub fn new_sdomain(dim: D) -> Self {
        Domain::SDomain(dim)
    }

    /// Create a new spherical shell domain centered at the origin with radius 1.
    pub fn new_shdomain(dim: D) -> Self {
        Domain::ShDomain(dim)
    }

    /// Returns the shape of the domain.
    pub fn shape_generic(&self) -> D {
        match self {
            Domain::UDomain(udom) => *udom,
            Domain::MDomain(sdoms) => sdoms.shape_generic().0,
            Domain::SDomain(dim) => *dim,
            Domain::ShDomain(dim) => *dim,
        }
    }

    /// Returns the size of the domain along each dimension.
    pub fn size(&self) -> OVector<Option<T>, D> {
        match self {
            Domain::UDomain(udom) => OVector::from_element_generic(*udom, U1, None),
            Domain::MDomain(sdoms) => OVector::from_iterator_generic(
                sdoms.shape_generic().0,
                U1,
                sdoms.iter().map(|sdom| match sdom {
                    (Some(min), Some(max)) => Some(max.clone() - min.clone()),
                    _ => None,
                }),
            ),
            Domain::SDomain(dim) => {
                OVector::from_element_generic(*dim, U1, Some(T::one() + T::one()))
            }
            Domain::ShDomain(dim) => {
                OVector::from_element_generic(*dim, U1, Some(T::one() + T::one()))
            }
        }
    }
}
