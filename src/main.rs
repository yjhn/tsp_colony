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

use crate::{
    arguments::{Args, PopulationSizes},
    benchmark::benchmark_ant_cycle,
    utils::Mpi,
};
use clap::Parser;
use mpi::{
    topology::{Process, SystemCommunicator},
    traits::{Communicator, Root},
};

// Run using:
// cargo build --release && RUST_BACKTRACE=1 mpirun -c 1 --use-hwthread-cpus --mca opal_warn_on_missing_libcuda 0 target/release/tsp_colony --dup switch-name --alphas 1 --betas 5 --ros 0.5 --qs 100 -f data/eil101.tsp 1>/dev/null
fn main() {
    let args = arguments::Args::parse();
    eprintln!("Supplied arguments: {args:#?}");

    let alphas = args.alphas.unwrap_or_else(|| vec![config::ALPHA]);
    let betas = args.betas.unwrap_or_else(|| vec![config::BETA]);
    let ros = args.ros.unwrap_or_else(|| vec![config::RO]);

    let capital_q_muls = args
        .capital_q_muls
        .unwrap_or_else(|| vec![config::CAPITAL_Q_MULTIPLIER]);
    let init_intensities = args
        .init_intensities
        .unwrap_or_else(|| vec![config::INITIAL_TRAIL_INTENSITY]);
    let lowercase_qs = args
        .lowercase_qs
        .unwrap_or_else(|| vec![config::LOWERCASE_Q]);
    let population_sizes = if let Some(popsizes) = args.population_sizes {
        PopulationSizes::Custom(popsizes)
    } else {
        PopulationSizes::SameAsCityCount
    };

    // Initialize MPI.
    const MPI_ROOT_RANK: i32 = 0;
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let world_size = world.size();
    let rank = world.rank();
    let root_process = world.process_at_rank(MPI_ROOT_RANK);
    let is_root = rank == MPI_ROOT_RANK;
    let mpi = Mpi {
        universe,
        world,
        world_size,
        root_process,
        rank,
        is_root: rank == MPI_ROOT_RANK,
    };

    benchmark_ant_cycle::<_, config::RNG>(
        &args.files,
        args.bench_repeat_times,
        population_sizes,
        args.max_iterations,
        args.dup,
        &args.bench_results_dir,
        &alphas,
        &betas,
        &capital_q_muls,
        &ros,
        &lowercase_qs,
        &init_intensities,
        &mpi,
    );
}
