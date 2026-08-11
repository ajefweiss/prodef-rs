# ProDeF - Probability Density Functions

A simple Rust crate for handling probability distributions. Provides efficient, type-safe APIs for evaluating, sampling, and manipulating probability densities across univariate, multivariate, and non-parametric representations.

All probability density functions (PDFs) implement the [`Density`] trait, the core trait of this crate, providing common operations like:
 - Evaluation: Compute probability density (non-normalized) for a given sample point
 - Sampling: Generate random samples according to the given distribution
 - Domain queries: Checking valid input ranges

The [`Domain`](domain::Domain) type defines the valid input space for any density function. 
A domain may be bounded, unbounded, or have a special structure. Currently only unbounded domains or hypercubes are supported.

## Distribution Types

- [`MultivariateDensity`](multivariate::MultivariateDensity) Combines multiple independent univariate distributions into a multivariate distribution. Use when the individual dimensions are statistically independent.
- [`MultivariateNormalDensity`](multinormal::MultivariateNormalDensity) A multivariate normal distribution with arbitrary covariance. Use when modeling correlated multi-dimensional data.
- [`ParticleDensity`](particle::ParticleDensity) A non-parametric density represented by weighted particles / samples. Use for complex distributions that can't be expressed analytically or for particle filters.
- [`KentDensity`](KentDensity) The Kent, or 5-parameter Fisher-Bingham, distribution that is defined on the surface of a unit sphere.

For the univariate case, one can use the following distributions directly:

- [`ConstantDensity`](ConstantDensity) - A degenerate distribution at a fixed value
- [`CosineDensity`](CosineDensity) - Cosine distribution over the interval [-π/2, π/2]
- [`LognormalDensity`](LognormalDensity) - Normal distribution in log-space (for positive values)
- [`LogUniformDensity`](LogUniformDensity) - Uniform distribution in log-space (for positive values)
- [`NormalDensity`](NormalDensity) - Normal distribution
- [`UniformDensity`](UniformDensity) - Uniform distribution over an interval

## Disclaimer

Some documentation and tests were created or enhanced using AI assistance. All code has been manually reviewed and verified to ensure correctness and adherence to project standards.
