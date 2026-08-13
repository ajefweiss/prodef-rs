use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::Vector3;
use prodef::{Density, KentDensity};
use rand::{SeedableRng, rngs::Xoshiro256PlusPlus};
use std::{hint::black_box, time::Duration};

fn bench_kent_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("kent_density");
    group.warm_up_time(Duration::new(0, 500_000_000));
    group.measurement_time(Duration::new(1, 0));

    let mu = Vector3::new(1.0, 0.0, 0.0);
    let g1 = Vector3::new(0.0, 1.0, 0.0);
    let dist = black_box(KentDensity::new(mu, g1, 2.0, 1.5));

    group.bench_function("density_at_mu", |b| {
        let dist_ref = &dist;
        b.iter(|| {
            let sample = mu;
            dist_ref.density::<nalgebra::U1, nalgebra::Const<3>>(&sample.as_view())
        })
    });

    group.finish();
}

fn bench_kent_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("kent_sample");
    group.warm_up_time(Duration::new(0, 500_000_000));
    group.measurement_time(Duration::new(3, 0));

    let mu = Vector3::new(1.0, 0.0, 0.0);
    let g1 = Vector3::new(0.0, 1.0, 0.0);
    let dist = black_box(KentDensity::new(mu, g1, 2.0, 1.5));

    group.throughput(criterion::Throughput::Elements(1));
    group.bench_function("sample_100", |b| {
        b.iter(|| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
            (0..100).map(|_| dist.sample(&mut rng)).last()
        })
    });

    group.finish();
}

criterion_group!(benches, bench_kent_density, bench_kent_sample);
criterion_main!(benches);
