mod normal;

use crate::Domain;
use nalgebra::{DefaultAllocator, Dim, Dyn, OMatrix, OVector, RealField, allocator::Allocator};
use rand::RngExt;
use rand_distr::{Uniform, uniform::SampleUniform};
use serde::{Deserialize, Serialize};

/// A non-parametric distribution defined by a population of `D`-dimensional particles.
///
/// The estimated density at a point x is computed using kernel density estimation with an adaptive
/// bandwidth matrix derived from the population covariance:
/// ```text
/// f̂(x) = (1/n) Σᵢ₌₁ⁿ K_H(x - xᵢ)  [unweighted]
/// f̂(x) = Σᵢ₌₁ⁿ wᵢ K_H(x - xᵢ)      [weighted]
/// ```
/// where xᵢ are particles, K_H is the kernel with bandwidth matrix H, and wᵢ are optional weights.
///
/// The generic kernel `K` is used for sampling. Sampling works by:
/// 1. Drawing a random particle from the population (uniformly if unweighted, or by weight if weighted)
/// 2. Sampling from the kernel centered at that particle
///
/// # Construction & Examples
///
/// Create a particle density with 100 random samples and Gaussian kernel:
/// ```
/// # use prodef::{ParticleDensity, MultivariateNormalDensity, Domain};
/// # use nalgebra::{OMatrix, U2, Dyn};
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
/// // Create particle density from matrix with automatic kernel fitting
/// let _density = ParticleDensity::<f64, U2, MultivariateNormalDensity<f64, U2>>::from_vectors::<Dyn, U2>(&particles.as_view(), domain, None, None);
/// ```
///
/// Create with weighted particles (importance sampling):
/// ```
/// # use prodef::{ParticleDensity, MultivariateNormalDensity, Domain};
/// # use nalgebra::{OMatrix, U2, Dyn, DVector};
/// # use rand::{SeedableRng, rngs::StdRng};
/// # use rand_distr::{Distribution, StandardNormal};
/// let mut rng = StdRng::seed_from_u64(42);
/// let n_samples = 50;
/// let mut particles = OMatrix::<f64, U2, Dyn>::zeros(n_samples);
/// for i in 0..n_samples {
///     particles[(0, i)] = StandardNormal.sample(&mut rng);
///     particles[(1, i)] = StandardNormal.sample(&mut rng);
/// }
/// // Equal weights
/// let weights: Vec<f64> = vec![1.0 / n_samples as f64; n_samples];
/// let domain = Domain::new_udomain(U2);
/// // Create particle density with weights from matrix (kernel fitted automatically)
/// let _density = ParticleDensity::<f64, U2, MultivariateNormalDensity<f64, U2>>::from_vectors::<Dyn, U2>(&particles.as_view(), domain, Some(&weights), None);
/// ```
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(bound(
    serialize = "D: Serialize, K: Serialize, OMatrix<T, D, Dyn>: Serialize, OVector<T, Dyn>: Serialize, Domain<T, D>: Serialize"
))]
#[serde(bound(
    deserialize = "D: Deserialize<'de>, K: Deserialize<'de>, OMatrix<T, D, Dyn>: Deserialize<'de>, OVector<T, Dyn>: Deserialize<'de>, Domain<T, D>: Deserialize<'de>"
))]
pub struct ParticleDensity<T, D, K>
where
    T: RealField,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, Dyn>,
{
    particles: OMatrix<T, D, Dyn>,
    opt_weights: Option<OVector<T, Dyn>>,
    domain: Domain<T, D>,
    kernel: K,
}

impl<T, D, K> ParticleDensity<T, D, K>
where
    T: RealField + SampleUniform,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, Dyn>,
{
    /// Returns the number of particles
    #[allow(clippy::len_without_is_empty)]
    pub fn count(&self) -> usize {
        self.particles.ncols()
    }

    /// Return a reference to the kernel density (used for sampling only).
    pub fn kernel(&self) -> &K {
        &self.kernel
    }

    /// Return a reference to the underlying particle matrix.
    pub fn particles(&self) -> &OMatrix<T, D, Dyn> {
        &self.particles
    }

    /// Draw a random sample particle.
    pub fn sample_particle(&self, rng: &mut impl RngExt) -> OVector<T, D> {
        let pdx = {
            match &self.opt_weights {
                Some(weights) =>
                // Here we abuse try_fold to return particle index early wrapped within Err().
                {
                    let uniform = Uniform::new(T::zero(), T::one()).unwrap();

                    // Select particle index by weight.
                    let wdx = rng.sample(uniform);

                    match weights
                        .iter()
                        .enumerate()
                        .try_fold(T::zero(), |acc, (idx, weight)| {
                            let next_weight = acc + weight.clone();
                            if wdx < next_weight {
                                Err(idx)
                            } else {
                                Ok(next_weight)
                            }
                        }) {
                        Ok(_) => weights.len() - 1,
                        Err(idx) => idx,
                    }
                }
                None => {
                    let uniform = Uniform::new(0, self.particles.ncols()).unwrap();

                    // Select particle index by weight.
                    rng.sample(uniform)
                }
            }
        };

        self.particles.column(pdx).clone_owned()
    }

    /// Return a reference to the particle weights.
    pub fn weights(&self) -> &Option<OVector<T, Dyn>> {
        &self.opt_weights
    }
}
