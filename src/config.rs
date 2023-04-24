/// Float type to use everywhere (pheromone and quantities).
pub type Float = f32;

// α: the relative importance of the trail, α ≥ 0
pub const ALPHA: Float = 0.5;
// β: the relative importance of the visibility, β ≥ 0
pub const BETA: Float = 0.5;
// ρ: trail persistence, 0≤ρ<1 (1-ρ can be interpreted as trail evaporation)
pub const PERSISTENCE: Float = 0.5;
// Q: a constant related to the quantity of trail laid by ants
pub const Q: Float = 0.5;

pub const POPULATION_SIZES: [u32; 5] = [8, 16, 32, 64, 128];
pub const MAX_ITERATIONS: u32 = 1000;
// Initial intensity of all trails.
pub const INITIAL_TRAIL_INTENSITY: Float = 0.2;

// Benchmark repeat times.
pub const REPEAT_TIMES: u32 = 10;
// Benchmark results directory.
pub const RESULTS_DIR: &str = "results";
