use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::{DMatrix, DVector, Dyn};
use prodef::{Density, Domain, MultivariateNormalDensity, ParticleDensity};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::{hint::black_box, time::Duration};

fn bench_particle_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("particle_density");
    group.warm_up_time(Duration::new(0, 500_000_000));
    group.measurement_time(Duration::new(1, 0));

    for dim in [2, 5, 10].iter() {
        let n_particles = 100;
        let particles = DMatrix::zeros(*dim, n_particles);
        let domain: Domain<f64, Dyn> = Domain::new_udomain(Dyn(*dim));
        let kernel_cov = DMatrix::identity(*dim, *dim) * 0.1;
        let kernel_mean = DVector::zeros(*dim);
        let kernel =
            MultivariateNormalDensity::new(kernel_cov, domain.clone(), Some(kernel_mean)).unwrap();
        let dist = black_box(
            ParticleDensity::from_vectors::<Dyn, Dyn>(
                &particles.as_view(),
                domain,
                None,
                Some(kernel),
            )
            .unwrap(),
        );

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

fn bench_particle_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("particle_sample");
    group.warm_up_time(Duration::new(0, 500_000_000));
    group.measurement_time(Duration::new(1, 0));

    for dim in [2, 5, 10].iter() {
        let n_particles = 100;
        let particles = DMatrix::zeros(*dim, n_particles);
        let domain: Domain<f64, Dyn> = Domain::new_udomain(Dyn(*dim));
        let kernel_cov = DMatrix::identity(*dim, *dim) * 0.1;
        let kernel_mean = DVector::zeros(*dim);
        let kernel =
            MultivariateNormalDensity::new(kernel_cov, domain.clone(), Some(kernel_mean)).unwrap();
        let dist = black_box(
            ParticleDensity::from_vectors::<Dyn, Dyn>(
                &particles.as_view(),
                domain,
                None,
                Some(kernel),
            )
            .unwrap(),
        );

        group.throughput(criterion::Throughput::Elements((*dim * 100) as u64));
        group.bench_function(format!("sample_dim_{}", dim), |b| {
            b.iter(|| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
                (0..100).map(|_| dist.sample(&mut rng)).last()
            })
        });
    }

    group.finish();
}

fn bench_particle_sample_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("particle_sample_iter");
    group.warm_up_time(Duration::new(0, 500_000_000));
    group.measurement_time(Duration::new(1, 0));

    for dim in [2, 5, 10].iter() {
        let n_particles = 100;
        let particles = DMatrix::zeros(*dim, n_particles);
        let domain: Domain<f64, Dyn> = Domain::new_udomain(Dyn(*dim));
        let kernel_cov = DMatrix::identity(*dim, *dim) * 0.1;
        let kernel_mean = DVector::zeros(*dim);
        let kernel =
            MultivariateNormalDensity::new(kernel_cov, domain.clone(), Some(kernel_mean)).unwrap();
        let dist = black_box(
            ParticleDensity::from_vectors::<Dyn, Dyn>(
                &particles.as_view(),
                domain,
                None,
                Some(kernel),
            )
            .unwrap(),
        );

        group.throughput(criterion::Throughput::Elements((*dim * 100) as u64));
        group.bench_function(format!("sample_iter_dim_{}", dim), |b| {
            b.iter(|| {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
                dist.sample_iter(&mut rng).take(100).flatten().last()
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_particle_density,
    bench_particle_sample,
    bench_particle_sample_iter
);
criterion_main!(benches);
