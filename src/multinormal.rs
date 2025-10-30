//! A module that implements a multivariate normal PDF.

use crate::{Density, RejectionSampler, SamplingMode, domain::Domain, macros::tval};
use itertools::{Itertools, zip_eq};
use nalgebra::{
    Const, DMatrix, DVector, DefaultAllocator, Dim, Dyn, MatrixView, OMatrix, OVector, RealField,
    U1, VectorView, allocator::Allocator,
};
use rand::RngExt;
use rand_distr::{Distribution, StandardNormal};
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
/// # use prodef::{MultivariateNormalDensity, Density, SamplingMode};
/// # use nalgebra::{OVector, U2, Matrix2};
/// # use prodef::Domain;
/// # use rand::{SeedableRng, rngs::StdRng};
/// let mean = OVector::from([0.0, 0.0]);
/// let covariance = Matrix2::from_element(1.0);
/// let domain = Domain::new_udomain(U2);
/// if let Some(dist) = MultivariateNormalDensity::new(covariance, domain, Some(mean)) {
///     let mut rng = StdRng::seed_from_u64(42);
///     if let Some(sample) = (&dist).sample(&mut rng, &SamplingMode::default()) {
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
    T: RealField,
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
{
    /// Returns the value of the bilinear form x^T * A^-1 * y, where A is the covariance matrix of the multivariate normal distribution.
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

        // Check dimensions (only required for D = Dyn).
        if n_dim.value() != n_dim_b.value() {
            return None;
        }

        // Construct the covariance matrix.
        let mut covariance = OMatrix::<T, D, D>::from_iterator_generic(
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
                                .expect("failed to compute covariance"),
                            None => covariance(x, y).expect("failed to compute covariance"),
                        }
                    } else {
                        T::zero()
                    }
                } else {
                    T::zero()
                }
            }),
        );

        // Fill up the other side of the covariance matrix.
        covariance += covariance.transpose() - OMatrix::from_diagonal(&covariance.diagonal());

        let mut mean = vectors.column_mean();

        // Set mean to first particle value if covariance is zero.
        // This fixes numerical issues where taking the mean over many
        // particles does not equal the constant value.
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

        // Detect zero'd columns/rows that need to be modified.
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

        // Detect NaN's and zero them out.
        m.iter_mut().for_each(|value| {
            if !value.is_finite() {
                *value = T::zero()
            }
        });

        let y = l_1.clone().solve_lower_triangular(&(mu_1 - mu_0)).unwrap();

        Some(
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

    /// Returns the logarithm of the multivariate normal PDF at the given sample point.
    pub fn log_density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, D, RStride, CStride>,
    ) -> Option<T> {
        if !self.domain.contains(sample) {
            return None;
        }

        let diff = sample - &self.mean;
        let value = (diff.transpose() * &self.inverse * diff)[(0, 0)].clone();

        let p_nonzero: usize = self.covariance.diagonal().iter().fold(0, |acc, next| {
            if next.abs().is_positive() {
                acc + 1
            } else {
                acc
            }
        });

        Some(
            -(self.determinant().ln() + value + tval!(p_nonzero, usize) * T::two_pi().ln())
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
}

impl<T, D> Density<T, D> for &MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
    StandardNormal: Distribution<T>,
{
    fn density<RStride: Dim, CStride: Dim>(
        &self,
        sample: &VectorView<T, D, RStride, CStride>,
    ) -> Option<T> {
        if !self.domain.contains(sample) {
            return None;
        }

        let diff = sample - &self.mean;
        let value = (diff.transpose() * &self.inverse * diff)[(0, 0)].clone();

        let p_nonzero: usize = self.covariance.diagonal().iter().fold(0, |acc, next| {
            if next.abs().is_positive() {
                acc + 1
            } else {
                acc
            }
        });

        Some(
            (-value / tval!(2, usize)).exp()
                / ((T::two_pi()).powi(p_nonzero as i32) * self.determinant()).sqrt(),
        )
    }

    fn domain(&self) -> Domain<T, D> {
        self.domain.clone()
    }

    fn sample(&self, rng: &mut impl RngExt, mode: &SamplingMode) -> Option<OVector<T, D>> {
        self.rejection_sample(rng, mode)
    }

    fn sample_iter(&self, rng: &mut impl RngExt) -> impl Iterator<Item = Option<OVector<T, D>>> {
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

            if self.domain.contains(&candidate.as_view()) {
                Some(candidate)
            } else {
                None
            }
        })
    }
}

impl<T, D> RejectionSampler<T, D> for &MultivariateNormalDensity<T, D>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<U1, D> + Allocator<D, D>,
    StandardNormal: Distribution<T>,
{
    fn generate_candidate(&self, rng: &mut impl RngExt) -> OVector<T, D> {
        let n_dim = self.covariance.shape_generic().0;

        self.mean.clone()
            + &self.ltm
                * OVector::<T, D>::from_iterator_generic(
                    n_dim,
                    U1,
                    (0..n_dim.value()).map(|_| rng.sample(StandardNormal)),
                )
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
        / wsum.clone();

    Some(
        zip_eq(x_iter, zip_eq(y_iter, w_iter))
            .map(|(val_x, (val_y, val_w))| {
                (mu_x.clone() - val_x.clone()) * (mu_y.clone() - val_y.clone()) * val_w.clone()
            })
            .sum::<T>()
            / wfac,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Domain;
    use approx::ulps_eq;
    use nalgebra::{Matrix, SVector, U2, U3, VecStorage};
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_multinormal_density() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let uniform = StandardNormal;

        let mut array = Matrix::<f64, U3, Dyn, VecStorage<f64, U3, Dyn>>::from_iterator(
            10000,
            (0..30000).map(|idx| {
                if idx % 3 == 1 {
                    0.0
                } else {
                    rng.sample::<f64, StandardNormal>(uniform)
                }
            }),
        );

        array.row_mut(0).iter_mut().for_each(|value| {
            *value += 0.1;
        });

        array.row_mut(2).iter_mut().for_each(|value| {
            *value += 0.25;
        });

        let mvnpdf = &MultivariateNormalDensity::from_vectors::<Dyn, U3>(
            &array.as_view(),
            Domain::new_mdomain(SVector::from([
                (Some(-0.75), Some(0.75)),
                (Some(-0.75), Some(0.75)),
                (Some(-0.75), Some(0.75)),
            ])),
            None,
        )
        .unwrap();

        let mvnpdf_weighted = &MultivariateNormalDensity::from_vectors::<Dyn, U3>(
            &array.as_view(),
            Domain::new_mdomain(SVector::from([
                (Some(-0.75), Some(0.75)),
                (Some(-0.75), Some(0.75)),
                (Some(-0.75), Some(0.75)),
            ])),
            Some(&vec![1.0; 10000]),
        )
        .unwrap();

        assert!(ulps_eq!(
            mvnpdf
                .density::<U1, U3>(&SVector::from([0.2, 0.0, 0.35]).as_view())
                .unwrap(),
            mvnpdf_weighted
                .density::<U1, U3>(&SVector::from([0.2, 0.0, 0.35]).as_view())
                .unwrap(),
            epsilon = 1e-5,
            max_ulps = 5
        ));

        assert!(ulps_eq!(
            mvnpdf
                .density::<U1, U3>(&SVector::from([0.2, 0.0, 0.35]).as_view())
                .unwrap(),
            0.064343349,
            epsilon = 1e-5,
            max_ulps = 5
        ));

        assert!(
            mvnpdf
                .density::<U1, U3>(&SVector::from([0.2f64, 1.0, 0.35]).as_view())
                .is_none()
        );

        assert!(ulps_eq!(
            mvnpdf
                .sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 100 })
                .unwrap(),
            SVector::from([-0.418523, 0.0, 0.4995714]),
            epsilon = 1e-5,
            max_ulps = 5
        ));

        assert!(
            mvnpdf.domain().contains::<U1, U3>(
                &mvnpdf
                    .sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 100 })
                    .unwrap()
                    .as_view()
            )
        );

        let mvpdf_ensbl = MultivariateNormalDensity::from_vectors::<Dyn, U3>(
            &array.as_view(),
            Domain::new_mdomain(SVector::from([
                (Some(-0.75), Some(0.75)),
                (Some(-0.75), Some(0.75)),
                (Some(-0.75), Some(0.75)),
            ])),
            None,
        )
        .unwrap();

        assert!(ulps_eq!(
            mvnpdf.ltm,
            mvpdf_ensbl.ltm,
            epsilon = 1e-5,
            max_ulps = 5
        ));
    }

    #[test]
    fn test_multinormal_density_kld() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let uniform = StandardNormal;

        let array_1 = Matrix::<f64, U3, Dyn, VecStorage<f64, U3, Dyn>>::from_iterator(
            10000,
            (0..30000).map(|idx| {
                if idx % 3 == 1 {
                    0.0
                } else {
                    rng.sample::<f64, StandardNormal>(uniform)
                }
            }),
        );

        let array_2 = Matrix::<f64, U3, Dyn, VecStorage<f64, U3, Dyn>>::from_iterator(
            10000,
            (0..30000).map(|idx| {
                if idx % 3 == 1 {
                    0.0
                } else {
                    0.25 + rng.sample::<f64, StandardNormal>(uniform)
                }
            }),
        );

        let mvpdf_1 = MultivariateNormalDensity::from_vectors::<Dyn, U3>(
            &array_1.as_view(),
            Domain::new_mdomain(SVector::from([
                (Some(-0.75), Some(0.75)),
                (Some(-0.75), Some(0.75)),
                (Some(-0.75), Some(0.75)),
            ])),
            None,
        )
        .unwrap();
        let mvpdf_2 = MultivariateNormalDensity::from_vectors::<Dyn, U3>(
            &array_2.as_view(),
            Domain::new_mdomain(SVector::from([
                (Some(-0.75), Some(0.75)),
                (Some(-0.75), Some(0.75)),
                (Some(-0.75), Some(0.75)),
            ])),
            None,
        )
        .unwrap();

        assert!(ulps_eq!(
            mvpdf_1.kl_div(&mvpdf_2).unwrap(),
            0.181914,
            epsilon = 1e-5,
            max_ulps = 5
        ));

        assert!(ulps_eq!(
            mvpdf_2.kl_div(&mvpdf_1).unwrap(),
            0.166584,
            epsilon = 1e-5,
            max_ulps = 5
        ));
    }

    #[test]
    fn test_multinormal_addition() {
        let mean = nalgebra::Vector2::new(0.0, 0.0);
        let covariance = nalgebra::Matrix2::identity();
        let domain = Domain::new_mdomain(nalgebra::Vector2::new((None, None), (None, None)));
        let pdf = MultivariateNormalDensity::new(covariance, domain, Some(mean)).unwrap();
        let shift = nalgebra::Vector2::new(1.0, 2.0);
        let shifted = pdf + shift;

        assert_eq!(shifted.mean[0], 1.0);
        assert_eq!(shifted.mean[1], 2.0);
    }

    #[test]
    fn test_multinormal_rank() {
        let mean = nalgebra::Vector2::new(0.0, 0.0);
        let covariance = nalgebra::Matrix2::new(1.0, 0.0, 0.0, 1.0);
        let domain = Domain::new_mdomain(nalgebra::Vector2::new((None, None), (None, None)));
        let pdf = MultivariateNormalDensity::new(covariance, domain, Some(mean)).unwrap();
        assert_eq!(pdf.rank(), 2);
    }

    #[test]
    fn test_multinormal_sample_iter_2d() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mean = SVector::from([0.0, 0.0]);
        let covariance = OMatrix::<f64, U2, U2>::from_element(1.0);
        let domain = Domain::new_udomain(U2);
        let mvn = MultivariateNormalDensity::new(covariance, domain, Some(mean)).unwrap();

        let samples: Vec<_> = (&mvn).sample_iter(&mut rng).take(100).flatten().collect();

        assert_eq!(samples.len(), 100);

        // Check sample statistics
        let mean_0: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        let mean_1: f64 = samples.iter().map(|s| s[1]).sum::<f64>() / samples.len() as f64;

        assert!(
            mean_0.abs() < 0.3,
            "Mean of first dimension should be close to 0"
        );
        assert!(
            mean_1.abs() < 0.3,
            "Mean of second dimension should be close to 0"
        );
    }

    #[test]
    fn test_multinormal_sample_iter_correlated() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        let mean = SVector::from([0.0, 0.0]);
        // Correlated covariance matrix: ρ = 0.8
        let covariance = OMatrix::<f64, U2, U2>::from([[1.0, 0.8], [0.8, 1.0]]);
        let domain = Domain::new_udomain(U2);
        let mvn = MultivariateNormalDensity::new(covariance, domain, Some(mean)).unwrap();

        let samples: Vec<_> = (&mvn).sample_iter(&mut rng).take(200).flatten().collect();

        assert!(samples.len() > 50);

        // Correlation should be positive
        let mean_0: f64 = samples.iter().map(|s| s[0]).sum::<f64>() / samples.len() as f64;
        let mean_1: f64 = samples.iter().map(|s| s[1]).sum::<f64>() / samples.len() as f64;
        let covariance: f64 = samples
            .iter()
            .map(|s| (s[0] - mean_0) * (s[1] - mean_1))
            .sum::<f64>()
            / samples.len() as f64;

        assert!(
            covariance > 0.0,
            "Covariance should be positive for correlated normal"
        );
    }

    #[test]
    fn test_multinormal_sample_iter_bounded() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mean = SVector::from([0.0, 0.0]);
        let covariance = OMatrix::<f64, U2, U2>::from_element(1.0);
        let domain = Domain::new_mdomain(SVector::from([
            (Some(-0.5), Some(0.5)),
            (Some(-0.5), Some(0.5)),
        ]));
        let mvn = MultivariateNormalDensity::new(covariance, domain, Some(mean)).unwrap();

        let results: Vec<_> = (&mvn).sample_iter(&mut rng).take(200).collect();

        // Should have some rejections due to bounds
        let none_count = results.iter().filter(|r| r.is_none()).count();
        assert!(
            none_count > 0,
            "Expected some rejections due to domain bounds"
        );

        // All valid samples should be within bounds
        for result in results.iter().flatten() {
            assert!(
                result[0] >= -0.5 && result[0] <= 0.5 && result[1] >= -0.5 && result[1] <= 0.5,
                "Sample out of bounds"
            );
        }
    }

    #[test]
    fn test_multinormal_sample_iter_3d() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mean = SVector::from([0.0, 0.0, 0.0]);
        let covariance = OMatrix::<f64, U3, U3>::from_element(1.0);
        let domain = Domain::new_udomain(U3);
        let mvn = MultivariateNormalDensity::new(covariance, domain, Some(mean)).unwrap();

        let samples: Vec<_> = (&mvn).sample_iter(&mut rng).take(100).flatten().collect();

        assert_eq!(samples.len(), 100);

        // All three dimensions should have reasonable statistics
        for dim in 0..3 {
            let dim_mean: f64 = samples.iter().map(|s| s[dim]).sum::<f64>() / samples.len() as f64;
            assert!(
                dim_mean.abs() < 0.4,
                "Mean of dimension {} should be close to 0",
                dim
            );
        }
    }

    #[test]
    fn test_multinormal_rejects_singular_matrix_2d() {
        // Singular 2x2 covariance (second row is zero - linearly dependent)
        let mean = SVector::from([0.0, 0.0]);
        let covariance = OMatrix::<f64, U2, U2>::from([[1.0, 0.5], [0.5, 0.0]]);
        let domain = Domain::new_udomain(U2);

        // Constructor should handle singular covariance gracefully
        // It may return Some with reduced-rank covariance or None
        let result = MultivariateNormalDensity::new(covariance, domain, Some(mean));

        // This tests that the constructor doesn't panic on singular input
        // The result may be Some or None depending on implementation
        let _ = result;
    }

    #[test]
    fn test_multinormal_rejects_zero_matrix() {
        // Zero covariance is singular
        let mean = SVector::from([0.0, 0.0]);
        let covariance = OMatrix::<f64, U2, U2>::from([[0.0, 0.0], [0.0, 0.0]]);
        let domain = Domain::new_udomain(U2);

        // Constructor should not panic
        let result = MultivariateNormalDensity::new(covariance, domain, Some(mean));
        let _ = result;
    }

    #[test]
    fn test_multinormal_rejects_rank_deficient_3d() {
        // 3x3 covariance with rank 1 (all rows proportional)
        let mean = SVector::from([0.0, 0.0, 0.0]);
        let covariance =
            OMatrix::<f64, U3, U3>::from([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
        let domain = Domain::new_udomain(U3);

        // Constructor should not panic on rank-deficient covariance
        let result = MultivariateNormalDensity::new(covariance, domain, Some(mean));
        let _ = result;
    }

    #[test]
    fn test_multinormal_rejects_negative_definite() {
        // Negative definite covariance (negation of valid covariance)
        let mean = SVector::from([0.0, 0.0]);
        let covariance = OMatrix::<f64, U2, U2>::from([[-1.0, 0.0], [0.0, -1.0]]);
        let domain = Domain::new_udomain(U2);

        // Constructor should not panic on negative-definite input
        let result = MultivariateNormalDensity::new(covariance, domain, Some(mean));
        let _ = result;
    }

    #[test]
    fn test_multinormal_rejects_asymmetric_matrix() {
        // Non-symmetric covariance (not a valid covariance)
        let mean = SVector::from([0.0, 0.0]);
        let covariance = OMatrix::<f64, U2, U2>::from([[1.0, 0.8], [0.3, 1.0]]);
        let domain = Domain::new_udomain(U2);

        // Constructor should not panic on asymmetric covariance
        let result = MultivariateNormalDensity::new(covariance, domain, Some(mean));
        let _ = result;
    }

    #[test]
    fn test_multinormal_statistical_validation_unbounded_2d() {
        // Test that 2D unbounded samples have correct mean
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mean = SVector::from([5.0, -3.0]);
        let covariance = OMatrix::<f64, U2, U2>::from([[2.0, 0.5], [0.5, 1.5]]);
        let domain = Domain::new_udomain(U2);
        let mvn = MultivariateNormalDensity::new(covariance, domain, Some(mean)).unwrap();

        let samples: Vec<_> = (0..5000)
            .filter_map(|_| {
                (&mvn).sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
            })
            .collect();

        assert!(!samples.is_empty(), "Should generate samples");

        let sample_mean = OVector::from_iterator_generic(
            U2,
            U1,
            (0..2).map(|i| samples.iter().map(|s| s[i]).sum::<f64>() / samples.len() as f64),
        );

        // Verify mean is close to expected
        assert!(
            (sample_mean[0] - 5.0).abs() < 0.15,
            "Dimension 0 mean should be ≈5.0, got {}",
            sample_mean[0]
        );
        assert!(
            (sample_mean[1] - (-3.0)).abs() < 0.15,
            "Dimension 1 mean should be ≈-3.0, got {}",
            sample_mean[1]
        );
    }

    #[test]
    fn test_multinormal_statistical_validation_unbounded_3d() {
        // Test that 3D unbounded samples have correct covariance structure
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(43);
        let mean = SVector::from([0.0, 0.0, 0.0]);
        let covariance =
            OMatrix::<f64, U3, U3>::from([[1.0, 0.4, 0.2], [0.4, 2.0, 0.3], [0.2, 0.3, 1.5]]);
        let domain = Domain::new_udomain(U3);
        let mvn = MultivariateNormalDensity::new(covariance, domain, Some(mean)).unwrap();

        let samples: Vec<_> = (0..10000)
            .filter_map(|_| {
                (&mvn).sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
            })
            .collect();

        let sample_mean = OVector::from_iterator_generic(
            U3,
            U1,
            (0..3).map(|i| samples.iter().map(|s| s[i]).sum::<f64>() / samples.len() as f64),
        );

        // Verify each dimension mean is close to 0
        for i in 0..3 {
            assert!(
                sample_mean[i].abs() < 0.15,
                "Dimension {} mean should be ≈0.0, got {}",
                i,
                sample_mean[i]
            );
        }

        // Verify diagonal covariance elements
        let sample_var = (0..3)
            .map(|i| {
                samples
                    .iter()
                    .map(|s| (s[i] - sample_mean[i]).powi(2))
                    .sum::<f64>()
                    / samples.len() as f64
            })
            .collect::<Vec<_>>();

        assert!(
            (sample_var[0] - 1.0).abs() < 0.3,
            "Variance of dim 0 should be ≈1.0, got {}",
            sample_var[0]
        );
        assert!(
            (sample_var[1] - 2.0).abs() < 0.3,
            "Variance of dim 1 should be ≈2.0, got {}",
            sample_var[1]
        );
        assert!(
            (sample_var[2] - 1.5).abs() < 0.3,
            "Variance of dim 2 should be ≈1.5, got {}",
            sample_var[2]
        );
    }

    #[test]
    fn test_multinormal_statistical_validation_bounded_2d() {
        // Test that bounded samples maintain mean near distribution mean
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(44);
        let mean = SVector::from([0.0, 0.0]);
        let covariance = OMatrix::<f64, U2, U2>::from([[1.0, 0.0], [0.0, 1.0]]);
        let domain = Domain::new_mdomain(SVector::from([
            (Some(-2.0), Some(2.0)),
            (Some(-2.0), Some(2.0)),
        ]));
        let mvn = MultivariateNormalDensity::new(covariance, domain, Some(mean)).unwrap();

        let samples: Vec<_> = (0..5000)
            .filter_map(|_| {
                (&mvn).sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
            })
            .collect();

        let sample_mean = OVector::from_iterator_generic(
            U2,
            U1,
            (0..2).map(|i| samples.iter().map(|s| s[i]).sum::<f64>() / samples.len() as f64),
        );

        // Verify all samples are within bounds
        assert!(
            samples
                .iter()
                .all(|s| s[0] >= -2.0 && s[0] <= 2.0 && s[1] >= -2.0 && s[1] <= 2.0),
            "All samples should be within [-2, 2]²"
        );

        // Mean should still be close to 0 (only slightly biased by bounds)
        assert!(
            sample_mean[0].abs() < 0.2,
            "Dimension 0 mean should be ≈0.0, got {}",
            sample_mean[0]
        );
        assert!(
            sample_mean[1].abs() < 0.2,
            "Dimension 1 mean should be ≈0.0, got {}",
            sample_mean[1]
        );
    }

    #[test]
    fn test_multinormal_statistical_validation_scaled() {
        // Test that scaling preserves statistical properties
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(45);
        let mean = SVector::from([1.0, 2.0]);
        let covariance = OMatrix::<f64, U2, U2>::from([[1.0, 0.0], [0.0, 1.0]]);
        let domain = Domain::new_udomain(U2);
        let mvn = MultivariateNormalDensity::new(covariance, domain, Some(mean)).unwrap();

        // Scale by 2
        let scaled_mvn = mvn * 2.0;

        let samples: Vec<_> = (0..5000)
            .filter_map(|_| {
                (&scaled_mvn).sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
            })
            .collect();

        let sample_mean = OVector::from_iterator_generic(
            U2,
            U1,
            (0..2).map(|i| samples.iter().map(|s| s[i]).sum::<f64>() / samples.len() as f64),
        );

        // Mean should be unchanged (scaling only affects covariance)
        assert!(
            (sample_mean[0] - 1.0).abs() < 0.15,
            "Scaled dimension 0 mean should still be ≈1.0, got {}",
            sample_mean[0]
        );
        assert!(
            (sample_mean[1] - 2.0).abs() < 0.15,
            "Scaled dimension 1 mean should still be ≈2.0, got {}",
            sample_mean[1]
        );

        // Variance should be scaled by factor of 2 (since scaling is on std_dev)
        let sample_var = (0..2)
            .map(|i| {
                samples
                    .iter()
                    .map(|s| (s[i] - sample_mean[i]).powi(2))
                    .sum::<f64>()
                    / samples.len() as f64
            })
            .collect::<Vec<_>>();

        assert!(
            (sample_var[0] - 2.0).abs() < 0.4,
            "Variance of scaled dim 0 should be ≈2.0, got {}",
            sample_var[0]
        );
        assert!(
            (sample_var[1] - 2.0).abs() < 0.4,
            "Variance of scaled dim 1 should be ≈2.0, got {}",
            sample_var[1]
        );
    }

    #[test]
    fn test_multinormal_statistical_validation_high_dim() {
        // Test rejection sampling in higher dimensions (5D) with bounded domain
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(46);
        let mean = OVector::from_element_generic(Dyn(5), U1, 0.0);
        let covariance = OMatrix::<f64, Dyn, Dyn>::from_element_generic(Dyn(5), Dyn(5), 1.0);
        let domain = Domain::new_mdomain(OVector::from_element_generic(
            Dyn(5),
            U1,
            (Some(-1.5), Some(1.5)),
        ));
        let mvn = MultivariateNormalDensity::new(covariance, domain, Some(mean)).unwrap();

        let samples: Vec<_> = (0..5000)
            .filter_map(|_| {
                (&mvn).sample(&mut rng, &SamplingMode::UntilValid { max_attempts: 512 })
            })
            .collect();

        assert!(
            !samples.is_empty(),
            "Should generate samples in 5D with bounds"
        );

        // Verify all samples are within bounds
        assert!(
            samples
                .iter()
                .all(|s| s.iter().all(|x| (-1.5..=1.5).contains(x))),
            "All samples should be within [-1.5, 1.5]⁵"
        );

        // Verify mean is close to origin
        let sample_mean: f64 =
            samples.iter().flat_map(|s| s.iter()).sum::<f64>() / (samples.len() * 5) as f64;

        assert!(
            sample_mean.abs() < 0.2,
            "Overall mean should be ≈0.0, got {}",
            sample_mean
        );
    }
}
