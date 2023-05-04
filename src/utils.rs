use crate::config::Float;
use crate::tour::CityIndex;
use mpi::environment::Universe;
use mpi::{
    topology::{Process, SystemCommunicator},
    traits::{Communicator, Root},
};

// Returns (low, high).
pub fn order<T: Ord>(a: T, b: T) -> (T, T) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

// Returns (high, low).
pub fn reverse_order<T: Ord>(a: T, b: T) -> (T, T) {
    if a > b {
        (a, b)
    } else {
        (b, a)
    }
}

pub fn maxf(x: Float, y: Float) -> Float {
    if x > y {
        x
    } else {
        y
    }
}

/// Creates a new `Vec<CityIndex>` containing all cities in order.
pub fn all_cities(count: u16) -> Vec<CityIndex> {
    (0..count).map(CityIndex::new).collect()
}

/// Fills up the buffer with cities in order.
/// Empties `buf` before proceeding.
/// It is neccessary to take `Vec` instead of slice,
/// since new elements need to be inserted.
pub fn all_cities_fill(buf: &mut Vec<CityIndex>, count: u16) {
    buf.clear();
    buf.extend((0..count).map(CityIndex::new));
}

/// Generates random seed and broadcasts it for every process.
pub fn initialize_random_seed(
    root_process: Process<SystemCommunicator>,
    rank: i32,
    is_root: bool,
) -> u64 {
    // Broadcast global random seed.
    let mut global_seed_buf = if is_root { [rand::random()] } else { [0] };
    root_process.broadcast_into(&mut global_seed_buf);
    global_seed_buf[0] + rank as u64
}

pub struct Mpi<'a> {
    pub universe: Universe,
    pub world: SystemCommunicator,
    pub world_size: i32,
    pub root_process: Process<'a, SystemCommunicator>,
    pub rank: i32,
    pub is_root: bool,
}

#[macro_export]
macro_rules! static_assert {
    ($cond:expr) => {
        $crate::static_assert!($cond, concat!("assertion failed: ", stringify!($cond)));
    };
    ($cond:expr, $($t:tt)+) => {
        const _: () = {
            if !$cond {
                panic!($($t)+)
            }
        };
    };
}
