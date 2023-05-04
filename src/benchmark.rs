use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{fmt::Display, path::Path};

use mpi::topology::{Process, SystemCommunicator};
use mpi::traits::{Communicator, Root};
use rand::{Rng, SeedableRng};
use serde::Serialize;

use crate::arguments::{DuplicateHandling, PopulationSizes};
use crate::config::Float;
use crate::paco_runner::PacoRunner;
use crate::tsp_problem::TspProblem;
use crate::utils::{initialize_random_seed, Mpi};

#[derive(Serialize)]
struct AntCycleConstants {
    max_iterations: u32,
    population_size: u32,
    alpha: Float,
    beta: Float,
    capital_q_mul: Float,
    ro: Float,
    lowercase_q: usize,
    init_g: u32,
    k: u32,
    init_intensity: Float,
}

#[derive(Serialize)]
struct RunResult {
    run_number: u32,
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

#[derive(Serialize)]
struct BenchmarkConfig<'a, T: Serialize> {
    process_count: i32,
    problem: ShortProblemDesc<'a>,
    algorithm: &'static str,
    algorithm_constants: T,
    repeat_times: u32,
}

/// Benchmark results are composed of a single identical configuration
/// run multiple times.
#[derive(Serialize)]
struct BenchmarkResults<'a, T: Serialize> {
    bench_config: BenchmarkConfig<'a, T>,
    benchmark_start_time_millis: u128,
    benchmark_duration_millis: u128,
    run_results: Vec<RunResult>,
}

pub fn benchmark_ant_cycle<PD, R>(
    problem_paths: &[PD],
    repeat_times: u32,
    population_sizes: PopulationSizes,
    max_iterations: u32,
    duplicate_handling: DuplicateHandling,
    results_dir: &str,
    alphas: &[Float],
    betas: &[Float],
    capital_q_multipliers: &[Float],
    ros: &[Float],
    lowercase_qs: &[usize],
    init_gs: &[u32],
    ks: &[u32],
    init_intensities: &[Float],
    mpi: &Mpi,
) where
    PD: AsRef<Path> + Display,
    R: Rng + SeedableRng,
{
    let process_count = mpi.world_size;
    let random_seed = initialize_random_seed(mpi.root_process, mpi.rank, mpi.is_root);
    let mut rng = R::seed_from_u64(random_seed);

    if mpi.is_root {
        // Create benchmark results directory.
        if !Path::new(results_dir).exists() {
            fs::create_dir(results_dir).unwrap();
        }

        eprintln!("MPI processes: {}", mpi.world_size);
    }

    for path in problem_paths {
        if mpi.is_root {
            eprintln!("Problem file name: {path}");
        }
        let problem = TspProblem::from_file(path);
        let population_sizes = match population_sizes {
            PopulationSizes::SameAsCityCount => vec![problem.number_of_cities() as u32],
            PopulationSizes::Custom(ref sizes) => sizes.clone(),
        };
        for &p in population_sizes.iter() {
            for &beta in betas {
                // Construct the solver here, as nothing meaningfull will change in
                // the inner loops.
                let mut ant_cycle = PacoRunner::new(
                    p as usize, &mut rng, &problem, 0.0, 0.0, beta, 0.0, 0.0, 0, 0, 0, mpi,
                );

                for &intense in init_intensities {
                    ant_cycle.reset_pheromone(intense);

                    for &alpha in alphas {
                        ant_cycle.set_alpha(alpha);

                        for &q in capital_q_multipliers {
                            ant_cycle.set_capital_q_mul(q);

                            for &ro in ros {
                                ant_cycle.set_ro(ro);

                                for &lowercase_q in lowercase_qs {
                                    ant_cycle.set_lowercase_q(lowercase_q);

                                    for &init_g in init_gs {
                                        ant_cycle.set_g(init_g);

                                        for &k in ks {
                                            ant_cycle.set_k(k);

                                            // Figure out where to save the results.
                                            let mut skip = [false];
                                            let save_file_path = if mpi.is_root {
                                                let mut save_file_path = format!(
                                        "{dir}/bm_{name}_{cpus}cpus_p{p}_q{q}_a{a}_b{b}_ro{r}_intensity{i}.json",
                                        dir=results_dir, name=problem.name(), cpus=process_count, p=p, q=q,
                                        a=alpha, b=beta, r=ro, i=intense
                                    );
                                                match get_output_file_path(
                                                    &mut save_file_path,
                                                    results_dir,
                                                    duplicate_handling,
                                                ) {
                                                    BenchAction::Continue => (),
                                                    BenchAction::SkipBench => skip[0] = true,
                                                    BenchAction::Abort => mpi.world.abort(1),
                                                };
                                                save_file_path
                                            } else {
                                                String::new()
                                            };
                                            mpi.root_process.broadcast_into(&mut skip);
                                            if skip[0] {
                                                continue;
                                            }

                                            // Run and time the benchmark.
                                            let mut run_results = Vec::new();
                                            let bench_start_absolute = SystemTime::now();
                                            let bench_start = Instant::now();

                                            let bench_config = BenchmarkConfig {
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
                                                    capital_q_mul: q,
                                                    ro,
                                                    max_iterations,
                                                    init_intensity: intense,
                                                    lowercase_q,
                                                    init_g,
                                                    k,
                                                },
                                                repeat_times,
                                            };

                                            if mpi.is_root {
                                                eprintln!(
                                                    "{}",
                                                    serde_json::to_string_pretty(&bench_config)
                                                        .unwrap()
                                                );
                                            }

                                            for run_number in 0..repeat_times {
                                                // for run_number in 0.. {
                                                let run_start = Instant::now();

                                                // todo!("Benchmark logic goes here");
                                                let found_optimal_tour =
                                                    ant_cycle.iterate_until_optimal(max_iterations);
                                                // Needed returns: found_optimal, shortest_found (per getter), iteration_reached (galima paimti per getter)
                                                if mpi.is_root {
                                                    let run_duration = run_start.elapsed();
                                                    let result = RunResult {
                                                        run_number,
                                                        found_optimal_tour,
                                                        shortest_found_tour: ant_cycle
                                                            .best_tour()
                                                            .length(),
                                                        iteration_reached: ant_cycle.iteration(),
                                                        duration_millis: run_duration.as_millis(),
                                                    };

                                                    // Dump partial results in case the benchmark is killed before completing all the runs.
                                                    eprintln!(
                                                        "{}",
                                                        serde_json::to_string_pretty(&result)
                                                            .unwrap()
                                                    );

                                                    run_results.push(result);
                                                }
                                                if found_optimal_tour {
                                                    break;
                                                }
                                                ant_cycle.reset_all_state();
                                            }

                                            let bench_duration = bench_start.elapsed();

                                            // Output results.
                                            if mpi.is_root {
                                                let results = BenchmarkResults {
                                                    bench_config,
                                                    benchmark_start_time_millis:
                                                        bench_start_absolute
                                                            .duration_since(UNIX_EPOCH)
                                                            .unwrap()
                                                            .as_millis(),
                                                    benchmark_duration_millis: bench_duration
                                                        .as_millis(),
                                                    run_results,
                                                };
                                                let json =
                                                    serde_json::to_string_pretty(&results).unwrap();
                                                // This is the actual meaningful output, so dump it to stdout
                                                // instead of stderr (allows to easily redirect it to file).
                                                println!("{json}");
                                                let mut file = open_output_file(
                                                    &save_file_path,
                                                    duplicate_handling,
                                                );
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
        eprint!("Found existing benchmark results file '{candidate_path}', ");
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
                // candidate_path.push_str(format!("{}", rand::random::<u32>()).as_str());
                candidate_path.push_str(&format!(
                    "{}",
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                ));
                eprintln!("will save new results to '{candidate_path}'");
            }
            // Handled later.
            DuplicateHandling::Overwrite => eprintln!("overwriting"),
            DuplicateHandling::Append => eprintln!("appending to it"),
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
