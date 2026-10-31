use crate::Density;
use log::warn;
use nalgebra::{DefaultAllocator, Dim, OVector, RealField, allocator::Allocator};
use rand::RngExt;
use rand_distr::{Distribution, StandardUniform};

/// A trait that implements the rejection sampling algorithm.
pub trait RejectionSampling<T, D>: Density<T, D>
where
    T: RealField,
    D: Dim,
{
    /// Generate a single candidate sample and return the sample along with the density for the proposal distribution.
    fn rejection_candidate<R>(&self, rng: &mut R) -> (OVector<T, D>, Option<T>)
    where
        R: RngExt,
        DefaultAllocator: Allocator<D>;

    /// Draw a random sample using rejection sampling.
    fn rejection_sample<R>(&self, rng: &mut R) -> OVector<T, D>
    where
        R: RngExt,
        DefaultAllocator: Allocator<D>,
        StandardUniform: Distribution<T>,
    {
        let mut attempts = 0;

        loop {
            let (candidate, proposal_density) = self.rejection_candidate(rng);

            if self.domain().contains(&candidate.as_view()) {
                // Special case if the candidate is within the domain but the returned proposal density is None.
                // Here we assume the target and proposal density are the same so we can automatically accept the candidate.
                if proposal_density.is_none() {
                    return candidate;
                }

                let target_density = self
                    .density(&candidate.as_view())
                    .expect("target density is invalid, this should not happen if the candidate is within the domain");
                let weight: T = rng.random();

                if weight * self.scale_factor()
                    <= target_density / proposal_density.expect("proposal density is invalid")
                {
                    return candidate;
                }
            }

            attempts += 1;

            if attempts == 1000 * self.ndim().pow(2) {
                warn!("rejection sampling exceeded {} attempts", attempts)
            }
        }
    }

    /// The scaling constant used in the rejection sampling algorithm.
    fn scale_factor(&self) -> T;
}
