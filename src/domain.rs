//! Types and traits for representing function domains.

use nalgebra::{
    DefaultAllocator, Dim, OVector, RealField, Scalar, U1, VectorView, allocator::Allocator,
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// A generic function domain specifying valid input regions for PDFs.
///
/// Two domain types are currently supported:
///
/// - **Unbounded (`UDomain`)**: The entire ℝᵈ space (all `D`-dimensional real values valid)
/// - **Bounded (`MDomain`)**: A `D`-dimensional hypercube with per-dimension interval bounds
///
/// # Operations
///
/// **Domain checking**: Use `contains()` to validate if a sample is in the domain.
/// - Returns `true` for samples within bounds (inclusive), `false` otherwise
/// - Used by `Density::density()` to return `None` for out-of-domain samples
/// - For bounded domains, returns `true` if `a ≤ sample ≤ b` in all dimensions
///
/// **Boundary enforcement**: Use `clamp()` to project samples onto domain boundaries.
/// - Clamps each coordinate to its dimension's [min, max] range (inclusive)
/// - Used by `SamplingMode::UntilValidOrClamp` after rejection sampling budget exhausted
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
#[serde(bound(serialize = "D: Serialize, OVector<(Option<T>, Option<T>), D>: Serialize"))]
#[serde(bound(
    deserialize = "D: Deserialize<'de>, OVector<(Option<T>, Option<T>), D>: Deserialize<'de>"
))]
pub enum Domain<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Unbounded domain spanning all of ℝᵈ (entire `D`-dimensional space).
    ///
    /// All samples are valid. Contains the dimension `D` for shape information.
    UDomain(D),
    /// Bounded domain as a `D`-dimensional hypercube with per-dimension interval bounds.
    ///
    /// Each element is a tuple `(min, max)` representing a **closed interval [min, max]**.
    /// - `Some(x)` means the bound is explicitly set to `x` (INCLUSIVE)
    /// - `None` means that bound is unbounded
    /// - `(None, None)` in a dimension is equivalent to unbounded in that coordinate
    /// - `(Some(a), Some(b))` is interpreted as the closed interval [a, b] where both endpoints are valid
    MDomain(OVector<(Option<T>, Option<T>), D>),
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

    /// Returns the shape of the domain.
    pub fn shape_generic(&self) -> D {
        match self {
            Domain::UDomain(udom) => *udom,
            Domain::MDomain(sdoms) => sdoms.shape_generic().0,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundaries_exclusive_outside() {
        // Verify that points just outside [0, 1] are NOT contained
        let domain = Domain::new_mdomain(OVector::from([(Some(0.0), Some(1.0))]));

        // Test just below lower bound (-0.001 should NOT be contained)
        let below_lower: f64 = -0.001;
        assert!(!domain.contains::<U1, U1>(&OVector::from([below_lower]).as_view()));

        // Test just above upper bound (1.001 should NOT be contained)
        let above_upper: f64 = 1.001;
        assert!(!domain.contains::<U1, U1>(&OVector::from([above_upper]).as_view()));

        // Test far outside bounds
        let far_negative: f64 = -1e6;
        assert!(!domain.contains::<U1, U1>(&OVector::from([far_negative]).as_view()));

        let far_positive: f64 = 1e6;
        assert!(!domain.contains::<U1, U1>(&OVector::from([far_positive]).as_view()));
    }

    #[test]
    fn test_boundaries_inclusive() {
        // Verify that boundary points [0, 1] are inclusive in bounded domain
        let domain = Domain::new_mdomain(OVector::from([(Some(0.0), Some(1.0))]));

        // Test lower boundary (0.0 should be contained)
        let lower_bound: f64 = 0.0;
        assert!(domain.contains::<U1, U1>(&OVector::from([lower_bound]).as_view()));

        // Test upper boundary (1.0 should be contained)
        let upper_bound: f64 = 1.0;
        assert!(domain.contains::<U1, U1>(&OVector::from([upper_bound]).as_view()));

        // Test interior point (0.5 should be contained)
        let interior: f64 = 0.5;
        assert!(domain.contains::<U1, U1>(&OVector::from([interior]).as_view()));
    }

    #[test]
    fn test_clamp_above_maximum() {
        let domain = Domain::new_mdomain(OVector::from([(Some(0.0), Some(1.0))]));
        let sample_above = OVector::from([1.5]);
        let clamped = domain.clamp::<U1, U1>(&sample_above.as_view());
        assert_eq!(clamped[0], 1.0);
    }

    #[test]
    fn test_clamp_below_minimum() {
        let domain = Domain::new_mdomain(OVector::from([(Some(0.0), Some(1.0))]));
        let sample_below = OVector::from([-0.5]);
        let clamped = domain.clamp::<U1, U1>(&sample_below.as_view());
        assert_eq!(clamped[0], 0.0);
    }

    #[test]
    fn test_clamp_half_bounded_lower() {
        // Test clamping with only lower bound (no upper bound)
        let domain = Domain::new_mdomain(OVector::from([(Some(0.0), None)]));

        // Value below lower bound should be clamped to lower bound
        let below = OVector::from([-1.0]);
        let clamped = domain.clamp::<U1, U1>(&below.as_view());
        assert_eq!(clamped[0], 0.0);

        // Value above lower bound should remain unchanged (no upper bound)
        let above = OVector::from([1e6]);
        let clamped = domain.clamp::<U1, U1>(&above.as_view());
        assert_eq!(clamped[0], 1e6);
    }

    #[test]
    fn test_clamp_half_bounded_upper() {
        // Test clamping with only upper bound (no lower bound)
        let domain = Domain::new_mdomain(OVector::from([(None, Some(1.0))]));

        // Value above upper bound should be clamped to upper bound
        let above = OVector::from([2.0]);
        let clamped = domain.clamp::<U1, U1>(&above.as_view());
        assert_eq!(clamped[0], 1.0);

        // Value below upper bound should remain unchanged (no lower bound)
        let below = OVector::from([-1e6]);
        let clamped = domain.clamp::<U1, U1>(&below.as_view());
        assert_eq!(clamped[0], -1e6);
    }

    #[test]
    fn test_clamp_unbounded() {
        // Test that clamping on unbounded domain returns sample unchanged
        let domain = Domain::new_udomain(U1);

        let sample = OVector::from([42.0]);
        let clamped = domain.clamp::<U1, U1>(&sample.as_view());
        assert_eq!(clamped[0], 42.0);
    }

    #[test]
    fn test_clamp_with_explicit_bounds() {
        // Test clamping with explicit lower and upper bounds
        let domain = Domain::new_mdomain(OVector::from([(Some(0.0), Some(1.0))]));

        // Value below lower bound should be clamped to lower bound
        let below = OVector::from([-0.5]);
        let clamped = domain.clamp::<U1, U1>(&below.as_view());
        assert_eq!(clamped[0], 0.0);

        // Value above upper bound should be clamped to upper bound
        let above = OVector::from([1.5]);
        let clamped = domain.clamp::<U1, U1>(&above.as_view());
        assert_eq!(clamped[0], 1.0);

        // Value within bounds should remain unchanged
        let inside = OVector::from([0.5]);
        let clamped = domain.clamp::<U1, U1>(&inside.as_view());
        assert_eq!(clamped[0], 0.5);
    }

    #[test]
    fn test_contains_unbounded() {
        let domain: Domain<f64, U1> = Domain::UDomain(U1);
        let sample = OVector::from([-1e6]);
        assert!(domain.contains::<U1, U1>(&sample.as_view()));
    }

    #[test]
    fn test_half_bounded_lower() {
        // Verify half-bounded domains work correctly (lower bound only, upper unbounded)
        // Note: lower bound is inclusive (value >= min)
        let domain = Domain::new_mdomain(OVector::from([(Some(0.0), None)]));

        // Exactly at lower bound should be contained (inclusive)
        assert!(domain.contains::<U1, U1>(&OVector::from([0.0]).as_view()));

        // Just above lower bound should be contained
        assert!(domain.contains::<U1, U1>(&OVector::from([1e-10]).as_view()));

        // Inside should be contained
        assert!(domain.contains::<U1, U1>(&OVector::from([1e6]).as_view()));

        // Below lower bound should NOT be contained
        assert!(!domain.contains::<U1, U1>(&OVector::from([-1e-10]).as_view()));
    }

    #[test]
    fn test_half_bounded_upper() {
        // Verify half-bounded domains work correctly (upper bound only, lower unbounded)
        // Note: upper bound is inclusive (value <= max)
        let domain = Domain::new_mdomain(OVector::from([(None, Some(1.0))]));

        // Exactly at upper bound should be contained (inclusive)
        assert!(domain.contains::<U1, U1>(&OVector::from([1.0]).as_view()));

        // Just below upper bound should be contained
        assert!(domain.contains::<U1, U1>(&OVector::from([1.0 - 1e-10]).as_view()));

        // Inside should be contained
        assert!(domain.contains::<U1, U1>(&OVector::from([-1e6]).as_view()));

        // Above upper bound should NOT be contained
        assert!(!domain.contains::<U1, U1>(&OVector::from([1.0 + 1e-10]).as_view()));
    }

    #[test]
    fn test_maximum_values() {
        let domain =
            Domain::new_mdomain(OVector::from([(Some(0.0), Some(1.0)), (Some(-5.0), None)]));
        let maxes = domain.maximum_values();
        assert_eq!(maxes[0], Some(1.0));
        assert_eq!(maxes[1], None);
    }
}
