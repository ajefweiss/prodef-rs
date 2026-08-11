use crate::Density;
use log::warn;
use nalgebra::{DefaultAllocator, Dim, OVector, RealField, allocator::Allocator};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Uniform, uniform::SampleUniform};

/// A trait that implements the Metropolis-Hastings sampling algorithm.
pub trait MetropolisHastingsSampling<T, D>: Density<T, D>
where
    T: RealField + SampleUniform,
    D: Dim,
{
    /// Generate a single candidate sample.
    fn mh_candidate<R>(&self, current: OVector<T, D>, rng: &mut R) -> OVector<T, D>
    where
        R: RngExt + SeedableRng,
        DefaultAllocator: Allocator<D>;

    /// Return an initial sample for the MH algorithm.
    fn mh_initialze(&self) -> OVector<T, D>
    where
        DefaultAllocator: Allocator<D>;

    /// Draw a random sample using the Metropolis-Hastings algorithm.
    fn mh_sample<R>(&self, rng: &mut R) -> OVector<T, D>
    where
        R: RngExt + SeedableRng,
        DefaultAllocator: Allocator<D>,
    {
        warn!(
            "individual sampling implementation using the metropolis-hastings is highly inefficient"
        );

        self.mh_sample_iter(rng).take(20).last().unwrap()
    }

    /// Iterator variant of [`Self::metropolis_hastings_sample`].
    ///
    /// This yields a stream of MH-updated states. For efficiency, the iterator caches the
    /// current state's `log_density` so it is not recomputed after every rejection.
    fn mh_sample_iter<R>(&self, rng: &mut R) -> impl Iterator<Item = OVector<T, D>>
    where
        T: SampleUniform,
        R: RngExt + SeedableRng,
        DefaultAllocator: Allocator<D>,
    {
        use std::iter::from_fn;

        // Cache the current state's log-density.
        let mut current = self.mh_initialze();
        let mut current_log_density = self.log_density(&current.as_view());
        let uniform = Uniform::new(T::zero(), T::one()).unwrap();

        from_fn(move || {
            let candidate = self.mh_candidate(current.clone_owned(), rng);
            let candidate_log_density = self.log_density(&candidate.as_view());

            // If candidate is out of domain => reject and return the current state unchanged.
            let Some(candidate_ld) = candidate_log_density else {
                return Some(current.clone_owned());
            };

            // If current is out of domain, accept the first valid candidate to recover from an invalid state.
            let next = match &current_log_density {
                None => {
                    current = candidate.clone_owned();
                    current_log_density = Some(candidate_ld);
                    current.clone_owned()
                }
                Some(cur_ld) => {
                    // alpha = min(1, exp(logf(candidate)-logf(current)))
                    let log_ratio = candidate_ld.clone() - cur_ld.clone();
                    let alpha = if log_ratio >= T::zero() {
                        T::one()
                    } else {
                        log_ratio.exp()
                    };

                    let accept = uniform.sample(rng) < alpha;
                    if accept {
                        current = candidate.clone_owned();
                        current_log_density = Some(candidate_ld);
                    }
                    current.clone_owned()
                }
            };

            Some(next)
        })
    }
}
