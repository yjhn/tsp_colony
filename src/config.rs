use rand::{
    rngs::{SmallRng, StdRng},
    CryptoRng,
};

// α: the relative importance of the trail, α ≥ 0
// TODO: remove this var as it's always 1, and raising to a power is very expensive.
pub const ALPHA: Float = 1.0;
// β: the relative importance of the visibility, β ≥ 0
pub const BETA: Float = 5.0;
// ρ: trail persistence, 0≤ρ<1 (1-ρ can be interpreted as trail evaporation)
pub const RO: Float = 0.5;
// Q: a constant related to the quantity of trail laid by ants
pub const CAPITAL_Q_MULTIPLIER: Float = 1.5;

pub const POPULATION_SIZES: [u32; 4] = [16, 32, 64, 128];
pub const MAX_ITERATIONS: u32 = 5000;
// Initial intensity of all trails.
pub const INITIAL_TRAIL_INTENSITY: Float = 100.0;
// q: how many most similar CPUs to take when calculating neighbour coefficient.
pub const LOWERCASE_Q: u16 = 3;

// Benchmark repeat times.
pub const REPEAT_TIMES: u32 = 10;
// Benchmark results directory.
pub const RESULTS_DIR: &str = "results";

/// Float type to use everywhere (pheromone and quantities).
pub type Float = f32;
// Alternatives: StdRng
pub type RNG = SmallRng;
