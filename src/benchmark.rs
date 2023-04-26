use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{fmt::Display, path::Path};

use mpi::topology::{Process, SystemCommunicator};
use mpi::traits::{Communicator, Root};
use rand::{Rng, SeedableRng};
use serde::Serialize;
use strum::IntoStaticStr;

use crate::ant_cycle::AntCycle;
use crate::arguments::DuplicateHandling;
use crate::config::Float;
use crate::tsp_problem::TspProblem;
use crate::utils::initialize_random_seed;

#[derive(Serialize)]
struct AntCycleConstants {
    max_iterations: u32,
    population_size: u32,
    alpha: Float,
    beta: Float,
    q: Float,
    ro: Float,
    init_intensity: Float,
}

#[derive(Serialize)]
struct RunResult {
    found_optimal_tour: bool,
    shortest_found_tour: u32,
    iteration_reached: u32,
    duration_millis: u128,
}

#[derive(Serialize)]
struct ShortProblemDesc<'a> {
    name: &'a str,
    optimal_length: u32,
}

/// Benchmark results are composed of a single identicla configuration
/// run multiple times.
#[derive(Serialize)]
struct BenchmarkResults<'a, T: Serialize> {
    benchmark_start_time_millis: u128,
    benchmark_duration_millis: u128,
    process_count: i32,
    problem: ShortProblemDesc<'a>,
    algorithm: &'static str,
    algorithm_constants: T,
    repeat_times: u32,
    run_results: Vec<RunResult>,
}

pub fn benchmark_ant_cycle<PD, R>(
    paths: &[PD],
    world: SystemCommunicator,
    root_process: Process<SystemCommunicator>,
    rank: i32,
    is_root: bool,
    repeat_times: u32,
    population_sizes: &[u32],
    max_iterations: u32,
    duplicate_handling: DuplicateHandling,
    results_dir: &str,
    alphas: &[Float],
    betas: &[Float],
    qs: &[Float],
    ros: &[Float],
    init_intensities: &[Float],
) where
    PD: AsRef<Path> + Display,
    R: Rng + SeedableRng,
{
    let process_count = world.size();
    let random_seed = initialize_random_seed(root_process, rank, is_root);
    let mut rng = R::seed_from_u64(random_seed);

    if is_root {
        // Create benchmark results directory.
        if !Path::new(results_dir).exists() {
            fs::create_dir(results_dir).unwrap();
        }

        eprintln!("MPI processes: {}", world.size());
        eprintln!("Population size: {population_sizes:?}");
    }

    for path in paths {
        if is_root {
            eprintln!("Problem file name: {path}");
        }
        let problem = TspProblem::from_file(path);
        for &p in population_sizes {
            for &alpha in alphas {
                for &beta in betas {
                    for &q in qs {
                        for &ro in ros {
                            for &intense in init_intensities {
                                let mut skip = [false];
                                let save_file_path = if is_root {
                                    let mut save_file_path = format!(
                                        "{dir}/bm_{name}_{cpus}cpus_p{p}_q{q}_a{a}_b{b}_ro{r}_intensity{i}.json",
                                        dir=results_dir, name=problem.name(), cpus=process_count, p=p, q=q, a=alpha, b=beta, r=ro, i=intense
                                    );
                                    match get_output_file_path(
                                        &mut save_file_path,
                                        results_dir,
                                        duplicate_handling,
                                    ) {
                                        BenchAction::Continue => (),
                                        BenchAction::SkipBench => skip[0] = true,
                                        BenchAction::Abort => world.abort(1),
                                    };
                                    save_file_path
                                } else {
                                    String::new()
                                };
                                root_process.broadcast_into(&mut skip);
                                if skip[0] {
                                    continue;
                                }

                                let mut run_results = Vec::new();
                                let bench_start_absolute = SystemTime::now();
                                let bench_start = Instant::now();
                                for run_number in 0..repeat_times {
                                    let run_start = Instant::now();

                                    // todo!("Benchmark logic goes here");

                                    if is_root {
                                        let run_duration = run_start.elapsed();
                                        let result = RunResult {
                                            found_optimal_tour: false,
                                            shortest_found_tour: 123456789,
                                            iteration_reached: max_iterations,
                                            duration_millis: run_duration.as_millis(),
                                        };
                                        run_results.push(result);
                                    }
                                }

                                let bench_duration = bench_start.elapsed();

                                // Output results.
                                if is_root {
                                    let results = BenchmarkResults {
                                        benchmark_start_time_millis: bench_start_absolute
                                            .duration_since(UNIX_EPOCH)
                                            .unwrap()
                                            .as_millis(),
                                        benchmark_duration_millis: bench_duration.as_millis(),
                                        process_count,
                                        problem: ShortProblemDesc {
                                            name: problem.name(),
                                            optimal_length: problem.solution_length(),
                                        },
                                        algorithm: "AntCycle",
                                        algorithm_constants: AntCycleConstants {
                                            population_size: p,
                                            alpha,
                                            beta,
                                            q,
                                            ro,
                                            max_iterations,
                                            init_intensity: intense,
                                        },
                                        repeat_times,
                                        run_results,
                                    };
                                    let json = serde_json::to_string_pretty(&results).unwrap();
                                    eprintln!("{json}");
                                    let mut file =
                                        open_output_file(&save_file_path, duplicate_handling);
                                    eprintln!("Saving results to '{save_file_path}'");
                                    file.write_all(json.as_bytes());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

enum BenchAction {
    Continue,
    SkipBench,
    Abort,
}

fn get_output_file_path(
    candidate_path: &mut String,
    results_dir: &str,
    duplicate_handling: DuplicateHandling,
) -> BenchAction {
    if Path::new(&candidate_path).exists() {
        eprint!("Existing benchmark results file '{candidate_path}' found, ");
        match duplicate_handling {
            DuplicateHandling::Panic => {
                eprintln!("terminating");
                return BenchAction::Abort;
            }
            DuplicateHandling::Skip => {
                eprintln!("skipping");
                return BenchAction::SkipBench;
            }

            DuplicateHandling::SwitchName => {
                candidate_path.push_str(format!("{}", rand::random::<u32>()).as_str());
                eprintln!("will save new results to '{candidate_path}'");
            }
            // Handled later.
            DuplicateHandling::Overwrite | DuplicateHandling::Append => (),
        };
        BenchAction::Continue
    } else {
        BenchAction::Continue
    }
}

fn open_output_file<P: AsRef<Path>>(path: P, duplicate_handling: DuplicateHandling) -> File {
    let mut openopts = OpenOptions::new();
    openopts.read(false).write(true);
    match duplicate_handling {
        DuplicateHandling::Overwrite => openopts.truncate(true).create(true),
        DuplicateHandling::Append => openopts.append(true).create(true),
        _ => openopts.create_new(true),
    };
    openopts.open(path).unwrap()
}
