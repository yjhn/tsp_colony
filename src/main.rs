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
    benchmark::{benchmark_ant_cycle, benchmark_qcabc},
    config::{paco, qcabc as abc},
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

    let alphas = args.alphas.unwrap_or_else(|| vec![paco::ALPHA]);
    let betas = args.betas.unwrap_or_else(|| vec![paco::BETA]);
    let ros = args.ros.unwrap_or_else(|| vec![paco::RO]);

    let capital_q_muls = args
        .capital_q_muls
        .unwrap_or_else(|| vec![paco::CAPITAL_Q_MULTIPLIER]);
    let init_intensities = args
        .init_intensities
        .unwrap_or_else(|| vec![paco::INITIAL_TRAIL_INTENSITY]);
    let lowercase_qs = args
        .lowercase_qs
        .unwrap_or_else(|| vec![config::LOWERCASE_Q]);
    let init_gs = args.init_gs.unwrap_or_else(|| vec![config::INIT_G]);
    let ks = args.ks.unwrap_or_else(|| vec![config::K]);
    let p_rcs = args.p_rcs.unwrap_or_else(|| vec![abc::P_RC]);
    let p_cps = args.p_cps.unwrap_or_else(|| vec![abc::P_CP]);
    let p_ls = args.p_ls.unwrap_or_else(|| vec![abc::P_L]);
    let l_mins = args.l_mins.unwrap_or_else(|| vec![abc::L_MIN]);
    let l_max_muls = args.l_max_muls.unwrap_or_else(|| vec![abc::L_MAX_MUL]);
    let nl_maxs = args.nl_maxs.unwrap_or_else(|| vec![abc::NL_MAX]);
    let rs = args.rs.unwrap_or_else(|| vec![abc::R]);
    let capital_ls = args.capital_ls.unwrap_or_else(|| abc::CAPITAL_LS.to_vec());

    let population_sizes = if let Some(popsizes) = args.population_sizes {
        PopulationSizes::Custom(popsizes)
    } else {
        PopulationSizes::SameAsCityCount
    };

    match args.algo {
        arguments::Algorithm::Paco => benchmark_ant_cycle::<_, config::MainRng>(
            &args.files,
            args.bench_repeat_times,
            population_sizes,
            args.max_iterations.unwrap_or(paco::MAX_ITERATIONS),
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
        ),
        arguments::Algorithm::Qcabc => benchmark_qcabc::<_, config::MainRng>(
            &args.files,
            &args.bench_results_dir,
            args.dup,
            args.bench_repeat_times,
            args.max_iterations.unwrap_or(abc::MAX_ITERATIONS),
            population_sizes,
            &nl_maxs,
            &p_cps,
            &p_rcs,
            &p_ls,
            &l_mins,
            &l_max_muls,
            &capital_ls,
            &rs,
            &lowercase_qs,
            &init_gs,
            &ks,
            &mpi,
        ),
    }
}
