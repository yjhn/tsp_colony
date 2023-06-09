use clap::{Parser, ValueEnum};

use crate::config::{self, paco, qcabc, Float};

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
    /// Append to old file. This should not be used, as JSON cannot have multiple roots.
    Append,
}

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(short, long, required = true, num_args(1..))]
    /// TSP problem definition files.
    pub files: Vec<String>,

    #[arg(long, required = false, default_value_t = config::RESULTS_DIR.to_owned())]
    pub bench_results_dir: String,

    #[arg(value_enum, long, required = false, default_value_t = DuplicateHandling::Panic)]
    /// What to do if existing benchmark results files are found.
    pub dup: DuplicateHandling,

    #[arg(value_enum, long, required = true)]
    pub algo: Algorithm,

    // Options applicable to both qCABC and PACO.
    #[arg(short, long, default_value_t = config::REPEAT_TIMES)]
    /// Number of times to repeat the benchmark.
    pub bench_repeat_times: u32,

    #[arg(long, num_args(1..))]
    pub lowercase_qs: Option<Vec<usize>>,

    #[arg(long, num_args(1..))]
    pub init_gs: Option<Vec<u32>>,

    #[arg(long, num_args(1..))]
    pub ks: Option<Vec<Float>>,

    #[arg(short, long)]
    /// Maximum number of generations for obtaining the optimal solution.
    pub max_iterations: Option<u32>,

    #[arg(short, long, num_args(1..))]
    /// Ant population sizes. If not specified, defaults to city count.
    pub population_sizes: Option<Vec<u32>>,

    #[arg(long, num_args(1..))]
    pub exchange_gens: Option<Vec<u32>>,

    // qCABC-specific options.
    #[arg(long, num_args(1..))]
    pub p_rcs: Option<Vec<Float>>,

    #[arg(long, num_args(1..))]
    pub p_cps: Option<Vec<Float>>,

    #[arg(long, num_args(1..))]
    pub p_ls: Option<Vec<Float>>,

    #[arg(long, num_args(1..))]
    pub l_mins: Option<Vec<usize>>,

    #[arg(long, num_args(1..))]
    pub l_max_muls: Option<Vec<Float>>,

    #[arg(long, num_args(1..))]
    pub nl_maxs: Option<Vec<u16>>,

    #[arg(long, num_args(1..))]
    pub rs: Option<Vec<Float>>,

    #[arg(long, num_args(1..))]
    pub capital_ls: Option<Vec<Float>>,

    // PACO-specific options.
    #[arg(long, num_args(1..))]
    pub alphas: Option<Vec<Float>>,

    #[arg(long, num_args(1..))]
    pub betas: Option<Vec<Float>>,

    #[arg(long, num_args(1..))]
    pub ros: Option<Vec<Float>>,

    #[arg(long, num_args(1..))]
    /// Multipliers for Q, which is related to the quantity of trail laid by ants.
    /// Q is calculated as capital_q_mul * \<largest length of the solutions in an iteration\>
    pub capital_q_muls: Option<Vec<Float>>,

    #[arg(long, num_args(1..))]
    pub init_intensities: Option<Vec<Float>>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum PopulationSizes {
    SameAsCityCount,
    Custom(Vec<u32>),
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, ValueEnum)]
pub enum Algorithm {
    Paco,
    Qcabc,
    Cabc,
}

impl Algorithm {
    pub fn as_str(self) -> &'static str {
        match self {
            Algorithm::Paco => "PACO",
            Algorithm::Qcabc => "qCABC",
            Algorithm::Cabc => "CABC",
        }
    }
}
