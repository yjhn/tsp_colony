#![allow(dead_code, unreachable_code, unused)]

mod ant;
mod ant_cycle;
mod arguments;
mod config;
mod distance_matrix;
mod matrix;
mod pheromone_visibility_matrix;
mod tour;
mod tsp_problem;
mod utils;

use crate::arguments::Args;
use clap::Parser;

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
}
