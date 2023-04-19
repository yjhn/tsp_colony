use clap::{Parser, ValueEnum};

use crate::config;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DuplicateHandling {
    /// Panic.
    Panic,
    /// Ignore (does not run duplicate benchmarks)
    Skip,
    /// Use a different name (append a number) for our benchamrk results file.
    SwitchName,
    /// Overwrite old file.
    Overwrite,
}

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(short, long, required = true, num_args(1..))]
    /// TSP problem definition files.
    pub files: Vec<String>,

    #[arg(long, required = false, default_value_t = config::ALPHA)]
    pub alpha: f32,

    #[arg(long, required = false, default_value_t = config::BETA)]
    pub beta: f32,

    #[arg(long, required = false, default_value_t = config::PERSISTENCE)]
    pub persistence: f32,

    #[arg(long, required = false, default_value_t = config::Q)]
    pub q: f32,

    #[arg(long, required = false, default_value_t = config::INITIAL_TRAIL_INTENSITY)]
    pub init_intensity: f32,

    #[arg(short, long, default_value_t = config::MAX_ITERATIONS)]
    /// Maximum number of generations for obtaining the optimal solution.
    pub max_iterations: u32,

    #[arg(short, long, default_value_t = config::REPEAT_TIMES)]
    /// Number of times to repeat the benchmark.
    pub bench_repeat_times: u32,

    #[arg(long, required = false, default_value_t = config::RESULTS_DIR.to_owned())]
    pub bench_results_dir: String,

    #[arg(short, long, required = false, num_args(1..))]
    pub population_sizes: Vec<u32>,

    #[arg(value_enum, long, required = false, default_value_t = DuplicateHandling::Panic)]
    /// What to do if existing benchmark results files are found.
    pub dup: DuplicateHandling,
}
