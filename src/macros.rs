macro_rules! tval {
    ($expr:expr, usize) => {
        T::from_usize($expr).unwrap()
    };
    ($expr:expr, f64) => {
        T::from_f64($expr).unwrap()
    };
}

pub(crate) use tval;
