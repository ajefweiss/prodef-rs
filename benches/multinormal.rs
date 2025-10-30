use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector, Dyn, OVector, U1};
use prodef::{Density, Domain, MultivariateNormalDensity, SamplingMode};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::{hint::black_box, time::Duration};

fn bench_multinormal_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("multinormal_density");
    group.warm_up_time(Duration::new(0, 500_000_000));
    group.measurement_time(Duration::new(1, 0));

    for dim in [2, 5, 10].iter() {
        let mean = DVector::zeros(*dim);
        let cov = DMatrix::identity(*dim, *dim);
        let domain: Domain<f64, Dyn> = Domain::new_udomain(Dyn(*dim));
        let dist =
            black_box(MultivariateNormalDensity::new(cov, domain, Some(mean.clone())).unwrap());

        group.throughput(criterion::Throughput::Elements(*dim as u64));
        group.bench_function(format!("density_dim_{}", dim), |b| {
            let dist_ref = &dist;
            b.iter(|| {
                let sample = DVector::zeros(*dim);
                dist_ref.density::<Dyn, Dyn>(&sample.as_view())
            })
        });
    }

    group.finish();
}

fn bench_multinormal_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("multinormal_sample");
    group.warm_up_time(Duration::new(0, 500_000_000));
    group.measurement_time(Duration::new(3, 0));

    for dim in [2, 5, 10].iter() {
        let mean = DVector::zeros(*dim);
        let cov = DMatrix::identity(*dim, *dim);
        let domain: Domain<f64, Dyn> = Domain::new_udomain(Dyn(*dim));
        let dist =
            black_box(MultivariateNormalDensity::new(cov, domain, Some(mean.clone())).unwrap());

        group.throughput(criterion::Throughput::Elements((*dim * 100) as u64));
        group.bench_function(format!("sample_dim_{}", dim), |b| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
            b.iter(|| {
                (0..100)
                    .map(|_| (&dist).sample(&mut rng, &SamplingMode::SingleAttempt))
                    .last()
            })
        });
    }

    group.finish();
}

fn bench_multinormal_sample_bounded(c: &mut Criterion) {
    let mut group = c.benchmark_group("multinormal_sample_bounded");
    group.warm_up_time(Duration::new(0, 500_000_000));
    group.measurement_time(Duration::new(3, 0));

    for dim in [2, 5, 10].iter() {
        let mean = DVector::zeros(*dim);
        let cov = DMatrix::identity(*dim, *dim);
        let domain: Domain<f64, Dyn> = Domain::new_mdomain(OVector::from_element_generic(
            Dyn(*dim),
            U1,
            (Some(-0.05), Some(0.05)),
        ));
        let dist =
            black_box(MultivariateNormalDensity::new(cov, domain, Some(mean.clone())).unwrap());

        group.throughput(criterion::Throughput::Elements((*dim * 100) as u64));
        group.bench_function(format!("sample_dim_{}", dim), |b| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
            b.iter(|| {
                (0..100)
                    .map(|_| (&dist).sample(&mut rng, &SamplingMode::SingleAttempt))
                    .last()
            })
        });
    }

    group.finish();
}

fn bench_multinormal_sample_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("multinormal_sample_iter");
    group.warm_up_time(Duration::new(0, 500_000_000));
    group.measurement_time(Duration::new(3, 0));

    for dim in [2, 5, 10].iter() {
        let mean = DVector::zeros(*dim);
        let cov = DMatrix::identity(*dim, *dim);
        let domain: Domain<f64, Dyn> = Domain::new_udomain(Dyn(*dim));
        let dist =
            black_box(MultivariateNormalDensity::new(cov, domain, Some(mean.clone())).unwrap());

        group.throughput(criterion::Throughput::Elements((*dim * 100) as u64));
        group.bench_function(format!("sample_iter_dim_{}", dim), |b| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
            b.iter(|| (&dist).sample_iter(&mut rng).take(100).flatten().last())
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_multinormal_density,
    bench_multinormal_sample,
    bench_multinormal_sample_bounded,
    bench_multinormal_sample_iter
);
criterion_main!(benches);
