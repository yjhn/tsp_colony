use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{fmt::Display, path::Path};

use mpi::topology::{Process, SystemCommunicator};
use mpi::traits::{Communicator, Root};
use rand::{Rng, SeedableRng};
use serde::Serialize;

use crate::arguments::{Algorithm, DuplicateHandling, PopulationSizes};
use crate::config::{DistanceT, Float};
use crate::parallel_ant_colony::PacoRunner;
use crate::qcabc::QuickCombArtBeeColony;
use crate::tsp_problem::TspProblem;
use crate::utils::{initialize_random_seed, IterateResult, Mpi};

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
    shortest_found_tour: DistanceT,
    iteration_reached: u32,
    shortest_iteration_tours: Vec<(u32, DistanceT)>,
    avg_iter_time_non_exch_micros: Float,
    avg_iter_time_exch_micros: Float,
    duration_millis: u128,
}

#[derive(Serialize)]
struct ShortProblemDesc<'a> {
    name: &'a str,
    optimal_length: DistanceT,
}

#[derive(Serialize)]
struct BenchmarkConfig<'a, T: Serialize> {
    process_count: usize,
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
    ks: &[Float],
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
                                            ant_cycle.set_k(k as u32);

                                            // Figure out where to save the results.
                                            let mut skip = [false];
                                            let save_file_path = if mpi.is_root {
                                                let mut save_file_path = format!(
                                        "{dir}/bm_paco_{name}_{cpus}cpus_p{p}_q{q}_a{a}_b{b}_ro{r}_intensity{i}.json",
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
                                                algorithm: "PACO",
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
                                                    k: k as u32,
                                                },
                                                repeat_times,
                                            };

                                            if mpi.is_root {
                                                println!(
                                                    "{}",
                                                    serde_json::to_string_pretty(&bench_config)
                                                        .unwrap()
                                                );
                                            }

                                            for run_number in 0..repeat_times {
                                                let run_start = Instant::now();

                                                let IterateResult {
                                                    found_optimal_tour,
                                                    shortest_iteration_tours,
                                                    avg_iter_time_non_exch_micros,
                                                    avg_iter_time_exch_micros,
                                                } = ant_cycle.iterate_until_optimal(max_iterations);
                                                if mpi.is_root {
                                                    let run_duration = run_start.elapsed();
                                                    let result = RunResult {
                                                        run_number,
                                                        found_optimal_tour,
                                                        shortest_iteration_tours,
                                                        shortest_found_tour: ant_cycle
                                                            .best_tour()
                                                            .length(),
                                                        iteration_reached: ant_cycle.iteration(),
                                                        duration_millis: run_duration.as_millis(),
                                                        avg_iter_time_non_exch_micros,
                                                        avg_iter_time_exch_micros,
                                                    };

                                                    // Dump partial results in case the benchmark is killed before completing all the runs.
                                                    println!(
                                                        "{}",
                                                        serde_json::to_string_pretty(&result)
                                                            .unwrap()
                                                    );

                                                    run_results.push(result);
                                                }
                                                // if found_optimal_tour {
                                                // break;
                                                // }
                                                ant_cycle.reset_all_state(init_g);
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

#[derive(Serialize)]
struct QcabcConstants {
    max_iterations: u32,
    colony_size: u32,
    nl_max: u16,
    p_cp: Float,
    p_rc: Float,
    p_l: Float,
    l_min: usize,
    l_max_mul: Float,
    r: Float,
    capital_l: Float,
    lowercase_q: usize,
    initial_g: u32,
    k: Float,
}

pub fn benchmark_qcabc<PD, R>(
    algo: Algorithm,
    problem_paths: &[PD],
    results_dir: &str,
    duplicate_handling: DuplicateHandling,
    repeat_times: u32,
    max_iterations: u32,
    colony_sizes: PopulationSizes,
    nl_maxs: &[u16],
    p_cps: &[Float],
    p_rcs: &[Float],
    p_ls: &[Float],
    l_mins: &[usize],
    l_max_muls: &[Float],
    capital_ls: &[Float],
    rs: &[Float],
    lowercase_qs: &[usize],
    initial_gs: &[u32],
    ks: &[Float],
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
        let colony_sizes = match colony_sizes {
            PopulationSizes::SameAsCityCount => vec![problem.number_of_cities() as u32],
            PopulationSizes::Custom(ref sizes) => sizes.clone(),
        };
        for colony_size in colony_sizes.iter().copied() {
            for nl_max in nl_maxs.iter().copied() {
                for p_cp in p_cps.iter().copied() {
                    for p_rc in p_rcs.iter().copied() {
                        for p_l in p_ls.iter().copied() {
                            for l_min in l_mins.iter().copied() {
                                for l_max_mul in l_max_muls.iter().copied() {
                                    for r in rs.iter().copied() {
                                        for lowercase_q in lowercase_qs.iter().copied() {
                                            for initial_g in initial_gs.iter().copied() {
                                                for k in ks.iter().copied() {
                                                    for capital_l in capital_ls.iter().copied() {
                                                        // Figure out where to save the results.
                                                        let mut skip = [false];
                                                        let save_file_path = if mpi.is_root {
                                                            let mut save_file_path = format!(
                                        "{dir}/bm_{algo}_{name}_{cpus}cpus_cs{cs}_nlmax{nlmax}_pcp{pcp}_prc{prc}_pl{pl}_lmin{lmin}_lmaxm{lmax}_r{r}_q{q}_g{g}_k{k}.json",
                                        algo=algo.as_str(), dir=results_dir, name=problem.name(), cpus=process_count,cs=colony_size, nlmax=nl_max, pcp=p_cp, prc=p_rc, pl=p_l, lmin=l_min, lmax=l_max_mul, r=r, q=lowercase_q, g=initial_g, k=k                                     );
                                                            match get_output_file_path(
                                                                &mut save_file_path,
                                                                results_dir,
                                                                duplicate_handling,
                                                            ) {
                                                                BenchAction::Continue => (),
                                                                BenchAction::SkipBench => {
                                                                    skip[0] = true
                                                                }
                                                                BenchAction::Abort => {
                                                                    mpi.world.abort(1)
                                                                }
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
                                                        let bench_start_absolute =
                                                            SystemTime::now();
                                                        let bench_start = Instant::now();

                                                        let bench_config = BenchmarkConfig {
                                                            process_count,
                                                            problem: ShortProblemDesc {
                                                                name: problem.name(),
                                                                optimal_length: problem
                                                                    .solution_length(),
                                                            },
                                                            algorithm: algo.as_str(),
                                                            algorithm_constants: QcabcConstants {
                                                                colony_size,
                                                                max_iterations,
                                                                lowercase_q,
                                                                initial_g,
                                                                k,
                                                                l_min,
                                                                l_max_mul,
                                                                p_cp,
                                                                p_rc,
                                                                p_l,
                                                                nl_max,
                                                                r,
                                                                capital_l,
                                                            },
                                                            repeat_times,
                                                        };

                                                        if mpi.is_root {
                                                            println!(
                                                                "{}",
                                                                serde_json::to_string_pretty(
                                                                    &bench_config
                                                                )
                                                                .unwrap()
                                                            );
                                                        }

                                                        for run_number in 0..repeat_times {
                                                            let mut qcabc =
                                                                QuickCombArtBeeColony::new(
                                                                    algo,
                                                                    &problem,
                                                                    colony_size,
                                                                    nl_max,
                                                                    capital_l,
                                                                    p_cp,
                                                                    p_rc,
                                                                    p_l,
                                                                    l_min,
                                                                    l_max_mul,
                                                                    r,
                                                                    lowercase_q,
                                                                    initial_g,
                                                                    k,
                                                                    &mut rng,
                                                                    mpi,
                                                                );
                                                            let run_start = Instant::now();

                                                            let IterateResult {
                                                                found_optimal_tour,
                                                                shortest_iteration_tours,
                                                                avg_iter_time_non_exch_micros,
                                                                avg_iter_time_exch_micros,
                                                            } = qcabc.iterate_until_optimal(
                                                                max_iterations,
                                                            );
                                                            if mpi.is_root {
                                                                let run_duration =
                                                                    run_start.elapsed();
                                                                let result = RunResult {
                                                                    run_number,
                                                                    found_optimal_tour,
                                                                    shortest_iteration_tours,
                                                                    shortest_found_tour: qcabc
                                                                        .best_tour()
                                                                        .length(),
                                                                    iteration_reached: qcabc
                                                                        .iteration(),
                                                                    duration_millis: run_duration
                                                                        .as_millis(),
                                                                    avg_iter_time_non_exch_micros,
                                                                    avg_iter_time_exch_micros,
                                                                };

                                                                // Dump partial results in case the benchmark is killed before completing all the runs.
                                                                println!(
                                                                    "{}",
                                                                    serde_json::to_string_pretty(
                                                                        &result
                                                                    )
                                                                    .unwrap()
                                                                );

                                                                run_results.push(result);
                                                            }
                                                            // if found_optimal_tour {
                                                            // break;
                                                            // }
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
                                                                benchmark_duration_millis:
                                                                    bench_duration.as_millis(),
                                                                run_results,
                                                            };
                                                            let json =
                                                                serde_json::to_string_pretty(
                                                                    &results,
                                                                )
                                                                .unwrap();
                                                            // This is the actual meaningful output, so dump it to stdout
                                                            // instead of stderr (allows to easily redirect it to file).
                                                            println!("{json}");
                                                            let mut file = open_output_file(
                                                                &save_file_path,
                                                                duplicate_handling,
                                                            );
                                                            eprintln!(
                                                            "Saving results to '{save_file_path}'"
                                                        );
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
        }
    }
}
