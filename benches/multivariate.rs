use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::U1;
use prodef::{
    ConstantDensity, CosineDensity, Density, LogUniformDensity, NormalDensity, SamplingMode,
    UniformDensity,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::{hint::black_box, time::Duration};

fn bench_multivariate_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("multivariate_density");
    group.warm_up_time(Duration::new(0, 200_000_000));
    group.measurement_time(Duration::new(0, 500_000_000));
    group.throughput(criterion::Throughput::Elements(1));

    // Constant
    let constant = black_box(ConstantDensity::new(0.5));
    group.bench_function("constant_density", |b| {
        let constant_ref = &constant;
        b.iter(|| {
            let sample = nalgebra::SVector::from([0.5]);
            constant_ref.density::<U1, U1>(&sample.as_view())
        })
    });

    // Normal
    let normal = black_box(NormalDensity::new(0.0, 1.0, None, None).unwrap());
    group.bench_function("normal_density", |b| {
        let normal_ref = &normal;
        b.iter(|| {
            let sample = nalgebra::SVector::from([0.5]);
            normal_ref.density::<U1, U1>(&sample.as_view())
        })
    });

    // Uniform
    let uniform = black_box(UniformDensity::new(0.0, 1.0).unwrap());
    group.bench_function("uniform_density", |b| {
        let uniform_ref = &uniform;
        b.iter(|| {
            let sample = nalgebra::SVector::from([0.5]);
            uniform_ref.density::<U1, U1>(&sample.as_view())
        })
    });

    // Cosine
    let cosine = black_box(CosineDensity::new(0.0, 1.0).unwrap());
    group.bench_function("cosine_density", |b| {
        let cosine_ref = &cosine;
        b.iter(|| {
            let sample = nalgebra::SVector::from([0.5]);
            cosine_ref.density::<U1, U1>(&sample.as_view())
        })
    });

    // Loguniform
    let loguniform = black_box(LogUniformDensity::new(0.1, 1.0).unwrap());
    group.bench_function("loguniform_density", |b| {
        let loguniform_ref = &loguniform;
        b.iter(|| {
            let sample = nalgebra::SVector::from([0.5]);
            loguniform_ref.density::<U1, U1>(&sample.as_view())
        })
    });

    group.finish();
}

fn bench_multivariate_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("multivariate_sample");
    group.warm_up_time(Duration::new(0, 200_000_000));
    group.measurement_time(Duration::new(0, 500_000_000));
    group.throughput(criterion::Throughput::Elements(1000));

    let constant = black_box(ConstantDensity::new(0.5));
    let normal = black_box(NormalDensity::new(0.0, 1.0, None, None).unwrap());
    let uniform = black_box(UniformDensity::new(0.0, 1.0).unwrap());
    let cosine = black_box(CosineDensity::new(0.0, 1.0).unwrap());
    let loguniform = black_box(LogUniformDensity::new(0.1, 1.0).unwrap());

    group.bench_function("constant_sample", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        b.iter(|| {
            (0..1000)
                .map(|_| constant.sample(&mut rng, &SamplingMode::SingleAttempt))
                .last()
        })
    });

    group.bench_function("normal_sample", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        b.iter(|| {
            (0..1000)
                .map(|_| normal.sample(&mut rng, &SamplingMode::SingleAttempt))
                .last()
        })
    });

    group.bench_function("uniform_sample", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        b.iter(|| {
            (0..1000)
                .map(|_| uniform.sample(&mut rng, &SamplingMode::SingleAttempt))
                .last()
        })
    });

    group.bench_function("cosine_sample", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        b.iter(|| {
            (0..1000)
                .map(|_| cosine.sample(&mut rng, &SamplingMode::SingleAttempt))
                .last()
        })
    });

    group.bench_function("loguniform_sample", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        b.iter(|| {
            (0..1000)
                .map(|_| loguniform.sample(&mut rng, &SamplingMode::SingleAttempt))
                .last()
        })
    });

    group.finish();
}

fn bench_multivariate_sample_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("multivariate_sample_iter");
    group.warm_up_time(Duration::new(0, 200_000_000));
    group.measurement_time(Duration::new(0, 500_000_000));
    group.throughput(criterion::Throughput::Elements(100));

    let constant = black_box(ConstantDensity::new(0.5));
    let normal = black_box(NormalDensity::new(0.0, 1.0, None, None).unwrap());
    let uniform = black_box(UniformDensity::new(0.0, 1.0).unwrap());
    let cosine = black_box(CosineDensity::new(0.0, 1.0).unwrap());
    let loguniform = black_box(LogUniformDensity::new(0.1, 1.0).unwrap());

    group.bench_function("constant_sample_iter", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        b.iter(|| constant.sample_iter(&mut rng).take(100).flatten().last())
    });

    group.bench_function("normal_sample_iter", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        b.iter(|| normal.sample_iter(&mut rng).take(100).flatten().last())
    });

    group.bench_function("uniform_sample_iter", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        b.iter(|| uniform.sample_iter(&mut rng).take(100).flatten().last())
    });

    group.bench_function("cosine_sample_iter", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        b.iter(|| cosine.sample_iter(&mut rng).take(100).flatten().last())
    });

    group.bench_function("loguniform_sample_iter", |b| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        b.iter(|| loguniform.sample_iter(&mut rng).take(100).flatten().last())
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_multivariate_density,
    bench_multivariate_sample,
    bench_multivariate_sample_iter
);
criterion_main!(benches);
