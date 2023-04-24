use std::fs::{self, OpenOptions};
use std::io::Write;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{fmt::Display, path::Path};

use mpi::topology::{Process, SystemCommunicator};
use mpi::traits::{Communicator, Root};
use rand::{Rng, SeedableRng};
use strum::IntoStaticStr;

use crate::ant_cycle::AntCycle;
use crate::arguments::DuplicateHandling;
use crate::config::Float;
use crate::tsp_problem::TspProblem;
use crate::utils::initialize_random_seed;

enum AlgorithmConstants {
    AntCycle {
        population_size: u32,
        alpha: Float,
        beta: Float,
        q: Float,
    },
}

struct RunResult {
    found_optimal_tour: bool,
    shortest_found_tour: u32,
    duration_millis: u128,
}

/// Benchmark results are composed of a single identicla configuration
/// run multiple times.
struct BenchmarkResults {
    benchmark_start_time: u128,
    benchmark_duration_millis: u128,
    process_count: i32,
    problem_name: String,
    algorithm_constants: AlgorithmConstants,
    repeat_times: u32,
    run_results: Vec<RunResult>,
}

pub fn benchmark_ant_cycle<PD, R>(
    path: PD,
    world: SystemCommunicator,
    root_process: Process<SystemCommunicator>,
    rank: i32,
    is_root: bool,
    repeat_times: u32,
    population_sizes: &[u32],
    exchange_generations: &[u32],
    duplicate_handling: DuplicateHandling,
    results_dir: &str,
    alphas: &[Float],
    betas: &[Float],
    qs: &[Float],
) where
    PD: AsRef<Path> + Display,
    R: Rng + SeedableRng,
{
    let process_count = world.size();

    if is_root {
        // Create benchmark results directory.
        if !Path::new(results_dir).exists() {
            fs::create_dir(results_dir).unwrap();
        }

        println!("Problem file name: {path}");
        println!("MPI processes: {}", world.size());
        println!("Population size: {population_sizes:?}");
    }

    // TODO: loop over multiple problem files.
    let problem = TspProblem::from_file(&path);
    for &p in population_sizes {
        for &alpha in alphas {
            for &beta in betas {
                for &q in qs {
                    let mut run_results = Vec::new();
                    let bench_start_absolute = SystemTime::now();
                    let bench_start = Instant::now();

                    for run_number in 0..repeat_times {
                        let run_start = Instant::now();

                        todo!("Benchmark logic goes here");

                        if is_root {
                            let run_duration = run_start.elapsed();
                            let result = RunResult {
                                found_optimal_tour: todo!(),
                                shortest_found_tour: todo!(),
                                duration_millis: run_duration.as_millis(),
                            };
                            run_results.push(result);
                        }
                    }

                    let bench_duration = bench_start.elapsed();

                    // Output results.
                    if is_root {
                        let save_file_path = format!(
                        "{results_dir}/bm_{}_{process_count}_cpus_p{p}_q{q}_al{alpha}_b{beta}.out", problem.name());

                        let results = BenchmarkResults {
                            benchmark_start_time: bench_start_absolute
                                .duration_since(UNIX_EPOCH)
                                .unwrap()
                                .as_millis(),
                            benchmark_duration_millis: bench_duration.as_millis(),
                            process_count,
                            problem_name: problem.name().to_owned(),
                            algorithm_constants: AlgorithmConstants::AntCycle {
                                population_size: p,
                                alpha,
                                beta,
                                q,
                            },
                            repeat_times,
                            run_results,
                        };
                        // TODO: serialize to JSON and dump to file
                    }
                }
            }
        }
    }
}
