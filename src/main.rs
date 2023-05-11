#![allow(dead_code, unreachable_code, unused, clippy::too_many_arguments)]

mod ant;
mod arguments;
mod benchmark;
mod config;
mod distance_matrix;
mod gstm;
mod index;
mod matrix;
mod parallel_ant_colony;
mod path_usage_matrix;
mod pheromone_visibility_matrix;
mod qcabc;
mod tour;
mod tsp_problem;
mod tsplib;
mod utils;

use std::process;

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
        world_size: world_size as usize,
        root_process,
        rank: rank as usize,
        is_root: rank == MPI_ROOT_RANK,
    };
    let args = if mpi.is_root {
        let args = arguments::Args::try_parse().unwrap_or_else(|e| {
            eprintln!("Error parsing arguments: {}", e);
            mpi.world.abort(2)
        });
        eprintln!("Supplied arguments: {args:#?}");
        args
    } else {
        arguments::Args::try_parse().unwrap_or_else(|_e| process::exit(2))
    };

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
    let init_gs = args.init_gs.unwrap_or_else(|| vec![config::INIT_G]);
    let ks = args.ks.unwrap_or_else(|| vec![config::K]);

    let population_sizes = if let Some(popsizes) = args.population_sizes {
        PopulationSizes::Custom(popsizes)
    } else {
        PopulationSizes::SameAsCityCount
    };

    benchmark_ant_cycle::<_, config::MainRng>(
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
        &init_gs,
        &ks,
        &init_intensities,
        &mpi,
    );
}
