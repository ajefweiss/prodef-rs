//! A module that implements a multivariate normal PDF.

use crate::{Density, domain::Domain, sampling::RejectionSampling, tval};
use itertools::{Itertools, zip_eq};
use nalgebra::{
    Const, DMatrix, DVector, DefaultAllocator, Dim, Dyn, MatrixView, OMatrix, OVector, RealField,
    Scalar, U1, VectorView, allocator::Allocator,
};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal, StandardUniform};
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    iter::{Sum, repeat_with},
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

/// A `D`-dimensional Gaussian distribution N(μ, Σ) with:
/// - **μ** (mean): Location vector in ℝᵈ
/// - **Σ** (covariance): Symmetric positive-definite d × d covariance
/// - Optional bounds: Hypercube [a₁, b₁] × ... × [aₐ, bₐ] for truncation
///
/// The density of this distribution is given by:
/// ```text
/// f(x) = (2π)^(-d/2) |Σ|^(-1/2) exp(-(1/2)(x - μ)ᵀ Σ⁻¹ (x - μ))
/// ```
///
/// # Construction & Examples
///
/// Correlated bivariate normal with ρ = 0.8:
/// ```
/// # use prodef::MultivariateNormalDensity;
/// # use nalgebra::{OVector, U2, Matrix2};
/// # use prodef::Domain;
/// let mean = OVector::from([0.0, 0.0]);
/// let covariance = Matrix2::from([
///     [1.0, 0.8],
///     [0.8, 1.0],
/// ]);
/// let domain = Domain::new_udomain(U2);
/// let _dist = MultivariateNormalDensity::new(covariance, domain, Some(mean));
/// ```
///
/// From empirical covariance (particles):
/// ```
/// # use prodef::MultivariateNormalDensity;
/// # use nalgebra::{OMatrix, U2, Dyn};
/// # use prodef::Domain;
/// # use rand::{SeedableRng, rngs::StdRng};
/// # use rand_distr::{Distribution, StandardNormal};
/// let mut rng = StdRng::seed_from_u64(42);
/// let n_samples = 100;
/// let mut particles = OMatrix::<f64, U2, Dyn>::zeros(n_samples);
/// for i in 0..n_samples {
///     particles[(0, i)] = StandardNormal.sample(&mut rng);
///     particles[(1, i)] = StandardNormal.sample(&mut rng);
/// }
/// let domain = Domain::new_udomain(U2);
/// let _dist = MultivariateNormalDensity::from_vectors::<Dyn, U2>(&particles.as_view(), domain, None);
/// ```
///
/// Evaluate density at a point:
/// ```
/// # use prodef::{MultivariateNormalDensity, Density};
/// # use nalgebra::{OVector, U1, U2, Matrix2};
/// # use prodef::Domain;
/// let mean = OVector::from([0.0, 0.0]);
/// let covariance = Matrix2::from_element(1.0);
/// let domain = Domain::new_udomain(U2);
/// if let Some(dist) = MultivariateNormalDensity::new(covariance, domain, Some(mean)) {
///     let sample = OVector::from([0.0, 0.0]);
///     if let Some(dens) = (&dist).density::<U1, U2>(&sample.as_view()) {
///         println!("Density at origin: {}", dens);
///     }
/// }
/// ```
///
/// Sample from the distribution:
/// ```
/// # use prodef::{MultivariateNormalDensity, Density};
/// # use nalgebra::{OVector, U2, Matrix2};
/// # use prodef::Domain;
/// # use rand::{SeedableRng, rngs::StdRng};
/// let mean = OVector::from([0.0, 0.0]);
/// let covariance = Matrix2::from_element(1.0);
/// let domain = Domain::new_udomain(U2);
/// if let Some(dist) = MultivariateNormalDensity::new(covariance, domain, Some(mean)) {
///     let mut rng = StdRng::seed_from_u64(42);
///     if let Some(sample) = (&dist).sample(&mut rng) {
///         println!("Generated sample: {:?}", sample);
///     }
/// }
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(
    serialize = "D: Serialize, OVector<T, D>: Serialize, OMatrix<T, D, D>: Serialize, Domain<T, D>: Serialize"
))]
#[serde(bound(
    deserialize = "D: Deserialize<'de>, OVector<T, D>: Deserialize<'de>, OMatrix<T, D, D>: Deserialize<'de>, Domain<T, D>: Deserialize<'de>"
))]
pub struct MultivariateNormalDensity<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    covariance: OMatrix<T, D, D>,
    inverse: OMatrix<T, D, D>,
    ltm: OMatrix<T, D, D>,
    domain: Domain<T, D>,

    /// The mean of the multivariate normal distribution.
    pub mean: OVector<T, D>,
}

impl<T, D> MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
    StandardUniform: Distribution<T>,
{
    /// Returns the value of the bilinear form x^T * A^-1 * y, where A is the covariance matrix of the multivariate normal distributio n.
    pub fn bilinear_map<RStride: Dim, CStride: Dim>(
        &self,
        x: &VectorView<T, D, RStride, CStride>,
        y: &VectorView<T, D, RStride, CStride>,
    ) -> T {
        (x.transpose() * &self.inverse * y)[(0, 0)].clone()
    }

    /// Returns a reference to the covariance matrix.
    pub fn covariance_matrix(&self) -> &OMatrix<T, D, D> {
        &self.covariance
    }

    /// Returns the determinant of the covariance matrix.
    ///
    /// The determinant is computed from the diagonal of the LTM decomposition. Since
    /// `ltm` stores `L * sqrt(D)` for a decomposed covariance `L * D * Lᵀ`, we square
    /// the product of the diagonal entries to recover |Σ|.
    pub fn determinant(&self) -> T {
        self.ltm
            .diagonal()
            .iter()
            .fold(T::one(), |acc, next| {
                if !next.is_zero() {
                    acc * next.clone()
                } else {
                    acc
                }
            })
            .powi(2)
    }

    /// Create a [`MultivariateNormalDensity`] from a set of vectors, with optional weights.
    pub fn from_vectors<RStride: Dim, CStride: Dim>(
        vectors: &MatrixView<T, D, Dyn, RStride, CStride>,
        domain: Domain<T, D>,
        opt_weights: Option<&[T]>,
    ) -> Option<Self>
    where
        T: Sum,
    {
        let n_dim = vectors.shape_generic().0;
        let n_dim_b = domain.shape_generic();
        let n_weights = match opt_weights {
            Some(weights) => weights.len(),
            None => vectors.ncols(),
        };

        // Check dimensions of the input vector matrix, the domain and the weights.
        if n_dim.value() != n_dim_b.value() || n_dim.value() == 0 || n_weights != vectors.ncols() {
            return None;
        }

        // Construct the covariance matrix.
        // Only compute the upper triangular half of the matrix here, then mirror it
        // to ensure symmetry. This avoids duplicate work for the covariance of each pair
        // of dimensions.
        let covariance_half = OMatrix::<T, D, D>::from_iterator_generic(
            n_dim,
            n_dim,
            (0..(vectors.nrows().pow(2))).map(|idx| {
                let jdx = idx / n_dim.value();
                let kdx = idx % n_dim.value();

                if jdx <= kdx {
                    let x = vectors.row(jdx);
                    let y = vectors.row(kdx);

                    if !x.iter().all_equal() && !y.iter().all_equal() {
                        match opt_weights {
                            Some(w) => covariance_with_weights(x, y, w)
                                .expect("covariance_with_weights was given a zero length vector"),
                            None => {
                                covariance(x, y).expect("covariance was given a zero length vector")
                            }
                        }
                    } else {
                        T::zero()
                    }
                } else {
                    T::zero()
                }
            }),
        );

        // Reflect the upper triangular part to the lower triangular side to obtain
        // a fully symmetric covariance matrix.
        let covariance = covariance_half.clone() + covariance_half.transpose()
            - OMatrix::from_diagonal(&covariance_half.diagonal());

        let mut mean = vectors.column_mean();

        // If any dimension has zero covariance, use the first particle in that
        // dimension as the mean. This stabilizes degenerate cases where the sample
        // covariance is exactly zero and the default mean estimate can be unreliable.
        covariance
            .diagonal()
            .iter()
            .zip(mean.iter_mut())
            .enumerate()
            .for_each(|(idx, (covariance, value))| {
                if covariance.is_zero() {
                    *value = vectors[(idx, 0)].clone();
                }
            });

        Self::new(covariance, domain, Some(mean))
    }

    /// Compute the Kullback-Leibler divergence between two [`MultivariateNormalDensity`]'s.
    pub fn kl_div(&self, other: &MultivariateNormalDensity<T, D>) -> Option<T>
    where
        T: Sum,
    {
        let mut l_0 = self.ltm.clone();
        let mu_0 = &self.mean;

        let mut l_1 = other.ltm.clone();
        let mu_1 = &other.mean;

        let mut n_dim = self.covariance.shape_generic().0.value();

        // For degenerate dimensions, replace zero diagonal values with infinity-like
        // entries and zero out the corresponding off-diagonals. This effectively
        // removes singular directions from the KL divergence contribution.
        (0..l_0.nrows()).for_each(|idx| {
            if l_0[(idx, idx)].is_zero() {
                l_0[(idx, idx)] = T::one() / T::zero();

                n_dim -= 1;

                // Set off diagonals to zero.
                for jdx in 0..l_0.ncols() {
                    if jdx != idx {
                        l_0[(idx, jdx)] = T::zero();
                        l_0[(jdx, idx)] = T::zero();
                    }
                }
            };
        });

        // Detect zero'd columns/rows that need to be modified.
        (0..l_1.nrows()).for_each(|idx| {
            if l_1[(idx, idx)].is_zero() {
                l_1[(idx, idx)] = T::one() / T::zero();

                // Set off diagonals to zero.
                for jdx in 0..l_1.ncols() {
                    if jdx != idx {
                        l_1[(idx, jdx)] = T::zero();
                        l_1[(jdx, idx)] = T::zero();
                    }
                }
            };
        });

        let mut m = l_1.clone().solve_lower_triangular(&l_0).unwrap();

        // Detect NaN's and zero them out so numerical instabilities do not propagate
        // into the final divergence value.
        m.iter_mut().for_each(|value| {
            if !value.is_finite() {
                *value = T::zero()
            }
        });

        let y = l_1.clone().solve_lower_triangular(&(mu_1 - mu_0)).unwrap();

        Some(
            // KL divergence of two multivariate normals:
            // 0.5 * [ tr(Σ1⁻¹ Σ0) - d + (μ1-μ0)ᵀ Σ1⁻¹ (μ1-μ0) + ln(|Σ1|/|Σ0|) ]
            // Here the computation is expressed in terms of the lower-triangular decompositions.
            (m.iter().cloned().sum::<T>() - tval!(n_dim, usize)
                + y.norm()
                + tval!(2, usize)
                    * l_1
                        .diagonal()
                        .iter()
                        .zip(l_0.diagonal().iter())
                        .map(|(a, b)| {
                            if a.is_finite() && b.is_finite() {
                                (a.clone() / b.clone()).ln()
                            } else {
                                T::zero()
                            }
                        })
                        .sum::<T>())
                / tval!(2, usize),
        )
    }

    /// Returns a reference to the LTM decomposition of the covariance matrix.
    pub fn lower_triangular_matrix(&self) -> &OMatrix<T, D, D> {
        &self.ltm
    }

    /// Returns the (squared) Mahalanobis distance.
    pub fn mahalanobis_distance_sq<RStride: Dim, CStride: Dim>(
        &self,
        x: &VectorView<T, D, RStride, CStride>,
    ) -> T {
        let xm = &(x - &self.mean);
        (xm.transpose() * &self.inverse * xm)[(0, 0)].clone()
    }

    /// Returns the normalization factor for the multivariate normal distribution.
    pub fn normalization_factor(&self) -> T {
        T::one()
            / (T::two_pi()).powf(tval!(self.rank(), usize) / tval!(2, usize))
            / self.determinant().sqrt()
    }

    /// Create a [`MultivariateNormalDensity`] from a covariance matrix.
    ///
    /// If no mean is provided, the mean is set to zero.
    pub fn new(
        covariance: OMatrix<T, D, D>,
        domain: Domain<T, D>,
        opt_mean: Option<OVector<T, D>>,
    ) -> Option<Self> {
        let n_dim = covariance.shape_generic().0;
        let n_dim_b = domain.shape_generic();

        let mean = match opt_mean {
            Some(mean) => mean.clone_owned(),
            None => OVector::<T, D>::zeros_generic(n_dim, Const::<1>),
        };

        // Check dimensions (only required for D = Dyn).
        if covariance.nrows() != mean.len() || n_dim.value() != n_dim_b.value() {
            return None;
        }

        let inverse = {
            // Convert input covariance into a DMatrix to bypass annoying ToTypenum domain.
            let dmatrix = DMatrix::from_iterator(
                covariance.nrows(),
                covariance.ncols(),
                covariance.iter().cloned(),
            );

            // Use a pseudo-inverse to support singular or nearly singular covariance matrices.
            // Later we explicitly zero any rows/columns in the pseudo-inverse corresponding to
            // exact zero variances to preserve degenerate subspaces.
            let mut pinv = dmatrix
                .clone_owned()
                .pseudo_inverse(T::default_epsilon())
                .expect("failed to compute pseudo inverse");

            dmatrix
                .diagonal()
                .iter()
                .enumerate()
                .for_each(|(idx, value)| {
                    if matches!(
                        value
                            .partial_cmp(&T::zero())
                            .expect("covariance matrix contains NaN values"),
                        Ordering::Equal
                    ) {
                        pinv.set_column(idx, &DVector::<T>::zeros(covariance.ncols()));
                        pinv.set_row(idx, &DVector::<T>::zeros(covariance.ncols()).transpose());
                    }
                });

            let n_dim = covariance.shape_generic().0;

            OMatrix::<T, D, D>::from_iterator_generic(n_dim, n_dim, pinv.iter().cloned())
        };

        let mut d = OVector::<T, D>::zeros_generic(n_dim, Const::<1>);
        let mut l = OMatrix::<T, D, D>::zeros_generic(n_dim, n_dim);

        // Compute an L D Lᵀ-style decomposition of the covariance matrix. Here `d` stores
        // the diagonal scaling terms and `l` stores the lower triangular factor.
        for cdx in 0..n_dim.value() {
            let mut d_j = covariance[(cdx, cdx)].clone();

            if cdx > 0 {
                for k in 0..cdx {
                    d_j -= d[k].clone() * l[(cdx, k)].clone().powi(2);
                }
            }

            d[cdx] = d_j;

            for rdx in cdx..n_dim.value() {
                let mut l_ij = covariance[(rdx, cdx)].clone();

                for k in 0..cdx {
                    l_ij -= d[k].clone() * l[(cdx, k)].clone() * l[(rdx, k)].clone();
                }

                if matches!(
                    d[cdx]
                        .partial_cmp(&T::zero())
                        .expect("covariance contains NaN values"),
                    Ordering::Equal
                ) {
                    l[(rdx, cdx)] = T::zero();
                } else {
                    l[(rdx, cdx)] = l_ij / d[cdx].clone();
                }
            }
        }

        let lsqrtd = l * OMatrix::from_diagonal(&OVector::from_iterator_generic(
            n_dim,
            Const::<1>,
            d.iter().map(|value| value.clone().sqrt()),
        ));

        // `lsqrtd` is the lower triangular factor multiplied by sqrt(diagonal scale),
        // which can be used to efficiently generate correlated normal samples.

        Some(Self {
            covariance,
            inverse,
            ltm: lsqrtd,
            domain,
            mean,
        })
    }

    /// Returns the rank of the underlying covariance matrix.
    pub fn rank(&self) -> usize {
        self.ltm
            .diagonal()
            .fold(0, |acc, next| if next != T::zero() { acc + 1 } else { acc })
    }

    // Set the variance of the selected dimension to zero.
    pub fn set_zero(&mut self, dim: usize) {
        self.covariance.column_mut(dim).fill(T::zero());
        self.covariance.row_mut(dim).fill(T::zero());

        self.inverse.column_mut(dim).fill(T::zero());
        self.inverse.row_mut(dim).fill(T::zero());

        self.ltm.column_mut(dim).fill(T::zero());
        self.ltm.row_mut(dim).fill(T::zero());
    }
}

impl<T, D> Density<T, D> for MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
    StandardNormal: Distribution<T>,
    StandardUniform: Distribution<T>,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, D, RStride, CStride>,
    ) -> Option<T> {
        if !self.domain.contains(sample) {
            return None;
        }

        Some(
            (-self.mahalanobis_distance_sq(sample) / tval!(2, usize)).exp()
                / ((T::two_pi()).powi(self.rank() as i32) * self.determinant()).sqrt(),
        )
    }

    fn domain(&self) -> Domain<T, D> {
        self.domain.clone()
    }

    fn log_density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, D, RStride, CStride>,
    ) -> Option<T> {
        if !self.domain.contains(sample) {
            return None;
        }

        Some(
            -(self.determinant().ln()
                + self.mahalanobis_distance_sq(sample)
                + tval!(self.rank(), usize) * T::two_pi().ln())
                / tval!(2, usize),
        )
    }

    fn mean(&self) -> OVector<T, D> {
        self.mean.clone()
    }

    fn sample<R>(&self, rng: &mut R) -> Option<OVector<T, D>>
    where
        R: RngExt + SeedableRng,
    {
        self.rejection_sample(rng)
    }

    fn sample_iter<R>(&self, rng: &mut R) -> impl Iterator<Item = Option<OVector<T, D>>>
    where
        R: RngExt + SeedableRng,
    {
        let normal = StandardNormal;
        let n_dim = self.covariance.shape_generic().0;

        repeat_with(move || {
            let candidate = self.mean.clone()
                + &self.ltm
                    * OVector::<T, D>::from_iterator_generic(
                        n_dim,
                        U1,
                        rng.sample_iter(normal).take(n_dim.value()),
                    );

            // Check if sample is within domain bounds
            if self.domain.contains(&candidate.as_view()) {
                Some(candidate)
            } else {
                None
            }
        })
    }

    fn variance(&self) -> OVector<T, D> {
        self.covariance_matrix().diagonal().clone_owned()
    }
}

impl<T, D> RejectionSampling<T, D> for &MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
    StandardNormal: Distribution<T>,
    StandardUniform: Distribution<T>,
{
    fn rejection_candidate<R>(&self, rng: &mut R) -> (OVector<T, D>, Option<T>)
    where
        R: RngExt + SeedableRng,
    {
        let n_dim = self.covariance.shape_generic().0;

        let candidate = self.mean.clone()
            + &self.ltm
                * OVector::<T, D>::from_iterator_generic(
                    n_dim,
                    U1,
                    (0..n_dim.value()).map(|_| rng.sample(StandardNormal)),
                );

        (candidate, None)
    }

    fn scale_factor(&self) -> T {
        T::one()
    }
}

impl<T, D> Add<OVector<T, D>> for MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    type Output = MultivariateNormalDensity<T, D>;

    fn add(self, rhs: OVector<T, D>) -> Self::Output {
        Self {
            covariance: self.covariance,
            inverse: self.inverse,
            ltm: self.ltm,
            domain: self.domain,
            mean: self.mean + rhs,
        }
    }
}

impl<T, D> Add<&OVector<T, D>> for MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    type Output = MultivariateNormalDensity<T, D>;

    fn add(self, rhs: &OVector<T, D>) -> Self::Output {
        Self {
            covariance: self.covariance,
            inverse: self.inverse,
            ltm: self.ltm,
            domain: self.domain,
            mean: self.mean + rhs,
        }
    }
}

impl<T, D> AddAssign<OVector<T, D>> for MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    fn add_assign(&mut self, rhs: OVector<T, D>) {
        self.mean += rhs
    }
}

impl<T, D> AddAssign<&OVector<T, D>> for MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    fn add_assign(&mut self, rhs: &OVector<T, D>) {
        self.mean += rhs
    }
}

impl<T, D> Mul<T> for MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    type Output = MultivariateNormalDensity<T, D>;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            covariance: self.covariance * rhs.clone(),
            inverse: self.inverse / rhs.clone(),
            ltm: self.ltm * rhs.sqrt(),
            domain: self.domain,
            mean: self.mean,
        }
    }
}

impl<T, D> MulAssign<T> for MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.covariance *= rhs.clone();
        self.inverse /= rhs.clone();
        self.ltm *= rhs.sqrt();
    }
}

impl<T, D> Sub<OVector<T, D>> for MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    type Output = MultivariateNormalDensity<T, D>;

    fn sub(self, rhs: OVector<T, D>) -> Self::Output {
        Self {
            covariance: self.covariance,
            inverse: self.inverse,
            ltm: self.ltm,
            domain: self.domain,
            mean: self.mean - rhs,
        }
    }
}

impl<T, D> Sub<&OVector<T, D>> for MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    type Output = MultivariateNormalDensity<T, D>;

    fn sub(self, rhs: &OVector<T, D>) -> Self::Output {
        Self {
            covariance: self.covariance,
            inverse: self.inverse,
            ltm: self.ltm,
            domain: self.domain,
            mean: self.mean - rhs,
        }
    }
}

impl<T, D> SubAssign<OVector<T, D>> for MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    fn sub_assign(&mut self, rhs: OVector<T, D>) {
        self.mean -= rhs
    }
}

impl<T, D> SubAssign<&OVector<T, D>> for MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
{
    fn sub_assign(&mut self, rhs: &OVector<T, D>) {
        self.mean -= rhs
    }
}

/// Computes the unbiased covariance over two slices.
///
/// The length of both iterators must be equal (panic).
pub fn covariance<'a, T, I>(x: I, y: I) -> Option<T>
where
    T: RealField + Sum,
    I: IntoIterator<Item = &'a T>,
    <I as IntoIterator>::IntoIter: Clone,
{
    let x_iter = x.into_iter();
    let y_iter = y.into_iter();

    let length = x_iter.clone().fold(0, |acc, _| acc + 1);

    if length == 0 {
        return None;
    }

    let mu_x = x_iter.clone().cloned().sum::<T>() / tval!(length, usize);
    let mu_y = y_iter.clone().cloned().sum::<T>() / tval!(length, usize);

    Some(
        zip_eq(x_iter, y_iter)
            .map(|(val_x, val_y)| (mu_x.clone() - val_x.clone()) * (mu_y.clone() - val_y.clone()))
            .sum::<T>()
            / tval!(length - 1, usize),
    )
}

/// Computes the unbiased covariance over two slices with weights.
///
/// The length of all three iterators must be equal (panic).
pub fn covariance_with_weights<'a, T, IV, IW>(x: IV, y: IV, w: IW) -> Option<T>
where
    T: RealField + Sum,
    IV: IntoIterator<Item = &'a T>,
    IW: IntoIterator<Item = &'a T>,
    <IV as IntoIterator>::IntoIter: Clone,
    <IW as IntoIterator>::IntoIter: Clone,
{
    let x_iter = x.into_iter();
    let y_iter = y.into_iter();
    let w_iter = w.into_iter();

    let wsum = w_iter.clone().cloned().sum::<T>();
    let wsumsq = w_iter.clone().map(|val_w| val_w.clone().powi(2)).sum::<T>();

    if wsum.is_zero() || w_iter.clone().any(|val_w| val_w.is_negative()) {
        return None;
    }

    let wfac = wsum.clone() - wsumsq / wsum.clone();

    let mu_x = zip_eq(x_iter.clone(), w_iter.clone())
        .map(|(val_x, val_w)| val_x.clone() * val_w.clone())
        .sum::<T>()
        / wsum.clone();

    let mu_y = zip_eq(y_iter.clone(), w_iter.clone())
        .map(|(val_y, val_w)| val_y.clone() * val_w.clone())
        .sum::<T>()
        / wsum;

    Some(
        zip_eq(x_iter, zip_eq(y_iter, w_iter))
            .map(|(val_x, (val_y, val_w))| {
                (mu_x.clone() - val_x.clone()) * (mu_y.clone() - val_y.clone()) * val_w.clone()
            })
            .sum::<T>()
            / wfac,
    )
}
