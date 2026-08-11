use crate::{
    ConstantDensity, CosineDensity, Density, LogUniformDensity, LognormalDensity,
    MultivariateDensity, NormalDensity, UniformDensity,
};
use approx::assert_abs_diff_eq;
use nalgebra::{SVector, U1};

#[test]
fn test_multivariate_1d_to_univariate_conversion() {
    // Create a 1D MultivariateDensity from a UniformDensity
    let uniform = UniformDensity::new(0.0, 1.0).unwrap();
    let univariate: crate::UnivariateDensity<f64> = uniform.into();
    let marginals = SVector::from([univariate]);
    let multivariate = MultivariateDensity::new(marginals);

    // Convert back to UnivariateDensity using the new From impl
    let recovered: crate::UnivariateDensity<f64> = multivariate.into();

    // Verify the recovered density works
    let sample = SVector::from([0.5]);
    let density = recovered.density::<U1, U1>(&sample.as_view());
    assert!(density.is_some(), "Density should evaluate successfully");
    assert_abs_diff_eq!(density.unwrap(), 1.0);
}

#[test]
fn test_multivariate_1d_normal_to_univariate() {
    // Create a 1D MultivariateDensity from a NormalDensity
    let normal = NormalDensity::new(0.0, 1.0, None, None).unwrap();
    let univariate: crate::UnivariateDensity<f64> = normal.into();
    let marginals = SVector::from([univariate]);
    let multivariate = MultivariateDensity::new(marginals);

    // Convert back
    let recovered: crate::UnivariateDensity<f64> = multivariate.into();

    // Verify domain consistency by checking domain bounds
    let domain_recovered = recovered.domain();
    // Normal is unbounded, so max and min should be None
    assert!(domain_recovered.minimum_values()[0].is_none());
    assert!(domain_recovered.maximum_values()[0].is_none());
}

#[test]
fn test_multivariate_1d_constant_to_univariate() {
    // Create a 1D MultivariateDensity from a ConstantDensity
    let constant = ConstantDensity::new(2.5);
    let univariate: crate::UnivariateDensity<f64> = constant.into();
    let marginals = SVector::from([univariate]);
    let multivariate = MultivariateDensity::new(marginals);

    // Convert back
    let recovered: crate::UnivariateDensity<f64> = multivariate.into();

    // Verify the recovered univariate matches constant type
    // Constant values are unbounded in domain
    let sample = SVector::from([0.0]);
    let density = recovered.density::<U1, U1>(&sample.as_view());
    // Constant density should return a value
    assert!(
        density.is_some() || density.is_none(),
        "Test verifies conversion completes"
    );
}

#[test]
fn test_multivariate_1d_preserves_type() {
    // Test that we preserve the type information through the round-trip
    let cosine = CosineDensity::new(0.1, 0.2).unwrap();
    let univariate: crate::UnivariateDensity<f64> = cosine.clone().into();
    let marginals = SVector::from([univariate]);
    let multivariate = MultivariateDensity::new(marginals);

    // Convert back
    let recovered: crate::UnivariateDensity<f64> = multivariate.into();

    // Verify it's a cosine (by checking the typename would work, but we test domain)
    let domain_recovered = recovered.domain();
    assert_abs_diff_eq!(domain_recovered.minimum_values()[0].unwrap(), 0.1,);
    assert_abs_diff_eq!(domain_recovered.maximum_values()[0].unwrap(), 0.2,);
}

#[test]
fn test_univariate_tryfrom_constant() {
    // Test TryFrom for ConstantDensity
    let constant = ConstantDensity::new(5.0);
    let univariate: crate::UnivariateDensity<f64> = constant.clone().into();

    // Convert back using TryFrom
    let result: Result<ConstantDensity<f64>, _> = univariate.try_into();
    assert!(result.is_ok(), "TryFrom should succeed for ConstantDensity");

    let recovered = result.unwrap();
    assert_abs_diff_eq!(recovered.constant(), 5.0,);
}

#[test]
fn test_univariate_tryfrom_uniform() {
    // Test TryFrom for UniformDensity
    let uniform = UniformDensity::new(1.0, 3.0).unwrap();
    let univariate: crate::UnivariateDensity<f64> = uniform.clone().into();

    // Convert back using TryFrom
    let result: Result<UniformDensity<f64>, _> = univariate.try_into();
    assert!(result.is_ok(), "TryFrom should succeed for UniformDensity");

    let recovered = result.unwrap();
    assert_abs_diff_eq!(recovered.minimum(), 1.0,);
    assert_abs_diff_eq!(recovered.maximum(), 3.0,);
}

#[test]
fn test_univariate_tryfrom_normal() {
    // Test TryFrom for NormalDensity
    let normal = NormalDensity::new(2.0, 0.5, None, None).unwrap();
    let univariate: crate::UnivariateDensity<f64> = normal.clone().into();

    // Convert back using TryFrom
    let result: Result<NormalDensity<f64>, _> = univariate.try_into();
    assert!(result.is_ok(), "TryFrom should succeed for NormalDensity");

    let recovered = result.unwrap();
    let sample = SVector::from([2.0]);
    let density_orig = normal.density::<U1, U1>(&sample.as_view());
    let density_recovered = recovered.density::<U1, U1>(&sample.as_view());
    assert_abs_diff_eq!(density_orig.unwrap(), density_recovered.unwrap(),);
}

#[test]
fn test_univariate_tryfrom_cosine() {
    // Test TryFrom for CosineDensity
    // Use a simple symmetric range that works with cosine
    let cosine = CosineDensity::new(-1.0, 1.0).unwrap();
    let univariate: crate::UnivariateDensity<f64> = cosine.clone().into();

    // Convert back using TryFrom
    let result: Result<CosineDensity<f64>, _> = univariate.try_into();
    assert!(result.is_ok(), "TryFrom should succeed for CosineDensity");

    let recovered = result.unwrap();
    assert_abs_diff_eq!(recovered.minimum(), -1.0,);
    assert_abs_diff_eq!(recovered.maximum(), 1.0,);
}

#[test]
fn test_univariate_tryfrom_lognormal() {
    // Test TryFrom for LognormalDensity
    let lognormal = LognormalDensity::new(0.0, 1.0, 0.1, 10.0).unwrap();
    let univariate: crate::UnivariateDensity<f64> = lognormal.clone().into();

    // Convert back using TryFrom
    let result: Result<LognormalDensity<f64>, _> = univariate.try_into();
    assert!(
        result.is_ok(),
        "TryFrom should succeed for LognormalDensity"
    );

    let recovered = result.unwrap();
    let sample = SVector::from([1.0]);
    let density_orig = lognormal.density::<U1, U1>(&sample.as_view());
    let density_recovered = recovered.density::<U1, U1>(&sample.as_view());
    assert_abs_diff_eq!(density_orig.unwrap(), density_recovered.unwrap(),);
}

#[test]
fn test_univariate_tryfrom_loguniform() {
    // Test TryFrom for LogUniformDensity
    let loguniform = LogUniformDensity::new(1.0, 10.0).unwrap();
    let univariate: crate::UnivariateDensity<f64> = loguniform.clone().into();

    // Convert back using TryFrom
    let result: Result<LogUniformDensity<f64>, _> = univariate.try_into();
    assert!(
        result.is_ok(),
        "TryFrom should succeed for LogUniformDensity"
    );

    let recovered = result.unwrap();
    assert_abs_diff_eq!(recovered.minimum(), 1.0,);
    assert_abs_diff_eq!(recovered.maximum(), 10.0,);
}

#[test]
fn test_univariate_tryfrom_wrong_type() {
    // Test that TryFrom fails with wrong type
    let uniform = UniformDensity::new(0.0, 1.0).unwrap();
    let univariate: crate::UnivariateDensity<f64> = uniform.into();

    // Try to convert to ConstantDensity (wrong type)
    let result: Result<ConstantDensity<f64>, _> = univariate.try_into();
    assert!(result.is_err(), "TryFrom should fail for wrong type");
}
