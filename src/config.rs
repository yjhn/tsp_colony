use rand::{
    rngs::{SmallRng, StdRng},
    CryptoRng,
};

pub mod paco {
    use super::Float;

    // α: the relative importance of the trail, α ≥ 0
    // TODO: remove this var as it's always 1, and raising to a power is very expensive.
    pub const ALPHA: Float = 1.0;
    // β: the relative importance of the visibility, β ≥ 0
    pub const BETA: Float = 5.0;
    // ρ: trail persistence, 0≤ρ<1 (1-ρ can be interpreted as trail evaporation)
    pub const RO: Float = 0.5;
    // Q: a constant related to the quantity of trail laid by ants
    pub const CAPITAL_Q_MULTIPLIER: Float = 1.5;

    // Initial intensity of all trails.
    pub const INITIAL_TRAIL_INTENSITY: Float = 100.0;

    pub const MIN_DELTA_TAU_INIT: f32 = 5.0;
}

pub const POPULATION_SIZES: [u32; 4] = [16, 32, 64, 128];
pub const MAX_ITERATIONS: u32 = 5000;
// q: how many most similar CPUs to take when calculating neighbour coefficient.
pub const LOWERCASE_Q: usize = 3;

pub const INIT_G: u32 = 1;

pub const K: u32 = 16;
// Benchmark repeat times.
pub const REPEAT_TIMES: u32 = 50;
// Benchmark results directory.
pub const RESULTS_DIR: &str = "results";

/// Float type to use everywhere (pheromone and quantities).
pub type Float = f32;

pub type DistanceT = u32;
pub trait Zeroable {
    const ZERO: Self;
}
impl Zeroable for DistanceT {
    const ZERO: Self = 0;
}
// Alternatives: StdRng
pub type MainRng = SmallRng;
