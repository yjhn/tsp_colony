#![allow(dead_code, unreachable_code, unused)]

mod ant;
mod ant_cycle;
mod arguments;
mod matrix;
mod pheromone_matrix;
mod tour;
mod tsp_problem;
mod utils;

use crate::arguments::Args;
use clap::Parser;

mod config {
    use rand::rngs::SmallRng;

    // α: the relative importance of the trail, α ≥ 0
    pub const ALPHA: f32 = 0.5;
    // β: the relative importance of the visibility, β ≥ 0
    pub const BETA: f32 = 0.5;
    // ρ: trail persistence, 0≤ρ<1 (1-ρ can be interpreted as trail evaporation)
    pub const PERSISTENCE: f32 = 0.5;
    // Q: a constant related to the quantity of trail laid by ants
    pub const Q: f32 = 0.5;

    pub const POPULATION_SIZES: [u32; 5] = [8, 16, 32, 64, 128];
    pub const MAX_ITERATIONS: u32 = 1000;
    // Initial intensity of all trails.
    pub const INITIAL_TRAIL_INTENSITY: f32 = 0.2;

    // Benchmark repeat times.
    pub const REPEAT_TIMES: u32 = 10;
    // Benchmark results directory.
    pub const RESULTS_DIR: &str = "results";

    pub type RNG = SmallRng;
}

fn main() {
    let args = arguments::Args::parse();
    dbg!(&args);
    let Args {
        files,
        alpha,
        beta,
        persistence,
        q,
        init_intensity,
        max_iterations,
        bench_repeat_times,
        bench_results_dir,
        population_sizes,
        dup,
    } = args;
    // We will use beta to raise the d_ij, not 1/d_ij,
    // so it must be negative to get the same results.
    let beta = -beta;
}
