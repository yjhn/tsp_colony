#![allow(dead_code, unreachable_code, unused, clippy::too_many_arguments)]

mod ant;
mod ant_cycle;
mod arguments;
mod benchmark;
mod config;
mod distance_matrix;
mod matrix;
mod pheromone_visibility_matrix;
mod tour;
mod tsp_problem;
mod tsplib;
mod utils;

use crate::{arguments::Args, benchmark::benchmark_ant_cycle};
use clap::Parser;
use mpi::{
    topology::{Process, SystemCommunicator},
    traits::{Communicator, Root},
};

// Run using:
// cargo build --release && RUST_BACKTRACE=1 mpirun -c 2 --use-hwthread-cpus --mca opal_warn_on_missing_libcuda 0 target/release/salesman -f data/att532.tsp -b 5 -a Cga --benchmark -m 100000 --bench-results-dir cga_100000_gens -e 4 --skip-duplicates
fn main() {
    let mut args = arguments::Args::parse();
    eprintln!("Supplied arguments: {args:#?}");

    if args.alphas.is_empty() {
        args.alphas = vec![config::ALPHA];
    }
    if args.betas.is_empty() {
        args.betas = vec![config::BETA];
    }
    if args.ros.is_empty() {
        args.ros = vec![config::RO];
    }
    if args.qs.is_empty() {
        args.qs = vec![config::Q];
    }
    if args.init_intensities.is_empty() {
        args.init_intensities = vec![config::INITIAL_TRAIL_INTENSITY];
    }
    if args.population_sizes.is_empty() {
        args.population_sizes = config::POPULATION_SIZES.to_vec();
    }

    eprintln!("Corrected arguments: {args:#?}");

    // Initialize stuff.
    const MPI_ROOT_RANK: i32 = 0;
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let size = world.size();
    let rank = world.rank();
    let root_process = world.process_at_rank(MPI_ROOT_RANK);
    let is_root = rank == MPI_ROOT_RANK;

    // TODO: bencmarking
    benchmark_ant_cycle::<_, config::RNG>(
        &args.files,
        world,
        root_process,
        rank,
        is_root,
        args.bench_repeat_times,
        &args.population_sizes,
        args.max_iterations,
        args.dup,
        &args.bench_results_dir,
        &args.alphas,
        &args.betas,
        &args.qs,
        &args.ros,
        &args.init_intensities,
    );
}
