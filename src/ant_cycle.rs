use std::{cmp::max, time::Instant};

use mpi::traits::CommunicatorCollectives;
use rand::{
    distributions::{self, Uniform},
    prelude::Distribution,
    seq::SliceRandom,
    Rng, SeedableRng,
};

use crate::{
    ant::Ant,
    config::Float,
    matrix::SquareMatrix,
    pheromone_visibility_matrix::PheromoneVisibilityMatrix,
    tour::{CityIndex, Tour, TourFunctions},
    tsp_problem::TspProblem,
    utils::Mpi,
};

/// Runs the ant cycle algorithm.
pub struct AntCycle<'a, R: Rng + SeedableRng> {
    /// Current time. Increases by number of cities in each iteration.
    // TODO: maybe this is not needed?
    time: usize,
    iteration: u32,
    ant_count: usize,
    ants: Vec<Ant>,
    rng: &'a mut R,
    best_tour: Tour,
    pheromone_matrix: PheromoneVisibilityMatrix,
    tsp_problem: &'a TspProblem,
    alpha: Float,
    capital_q_mul: Float,
    initial_trail_intensity: Float,
    lowercase_q: u16,
    mpi: &'a Mpi<'a>,
    cities_distrib: Uniform<u16>,
}

impl<'a, R: Rng + SeedableRng> AntCycle<'a, R> {
    pub fn new(
        ant_count: usize,
        mut rng: &'a mut R,
        tsp_problem: &'a TspProblem,
        initial_trail_intensity: Float,
        alpha: Float,
        beta: Float,
        capital_q_mul: Float,
        ro: Float,
        lowercase_q: u16,
        mpi: &'a Mpi,
    ) -> AntCycle<'a, R> {
        let city_count = tsp_problem.number_of_cities();
        let pheromone_matrix = PheromoneVisibilityMatrix::new(
            city_count as usize,
            initial_trail_intensity,
            tsp_problem.distances(),
            beta,
            ro,
        );
        let mut ants = Vec::with_capacity(ant_count);

        let distrib = distributions::Uniform::new(0, city_count)
            .unwrap()
            .sample_iter(&mut rng)
            .map(CityIndex::new);
        for random_city in distrib.take(ant_count) {
            let ant = Ant::new(city_count, random_city);
            ants.push(ant);
        }

        AntCycle {
            time: 0,
            iteration: 0,
            ant_count,
            ants,
            rng,
            best_tour: Tour::PLACEHOLDER,
            pheromone_matrix,
            tsp_problem,
            alpha,
            capital_q_mul,
            initial_trail_intensity,
            lowercase_q,
            mpi,
            cities_distrib: distributions::Uniform::new(0, city_count).unwrap(),
        }
    }

    pub fn set_alpha(&mut self, alpha: Float) {
        self.alpha = alpha;
    }

    pub fn set_capital_q_mul(&mut self, q: Float) {
        self.capital_q_mul = q;
    }

    pub fn set_ro(&mut self, ro: Float) {
        self.pheromone_matrix.set_ro(ro);
    }

    /// Resets pheromone intensity to `intensity`.
    pub fn reset_pheromone(&mut self, intensity: Float) {
        self.pheromone_matrix.reset_pheromone(intensity);
    }

    pub fn reset_all_state(&mut self) {
        // Reset ants.
        let num_cities = self.number_of_cities();
        for a in self.ants.iter_mut() {
            a.reset_to_city(
                num_cities,
                CityIndex::new(self.cities_distrib.sample(&mut self.rng)),
            );
        }
        self.time = 0;
        self.iteration = 0;
        self.best_tour = Tour::PLACEHOLDER;
        self.pheromone_matrix
            .reset_pheromone(self.initial_trail_intensity);
    }

    pub fn iterate_until_optimal(&mut self, max_iterations: u32) -> bool {
        let num_cities = self.number_of_cities() as usize;
        let distrib01 = distributions::Uniform::new(0.0, 1.0).unwrap();

        let mut found_optimal = false;
        let mut delta_tau_matrix = SquareMatrix::new(num_cities, 0.0);

        let world_size = self.mpi.world_size as usize;
        let mut cpu_random_order: Vec<usize> = (0..world_size).collect();
        // We will never exchange with ourselves.
        cpu_random_order.swap_remove(self.mpi.rank as usize);
        // Buffer for CPUs' best tours.
        // Tour::APPENDED_HACK_ELEMENTS extra spaces at the end ar for tour length and MPI rank.
        let mut cpus_best_tours_buf =
            vec![CityIndex::new(0); world_size * (num_cities + Tour::APPENDED_HACK_ELEMENTS)];
        let mut proc_distances = SquareMatrix::new(world_size, 0);

        loop {
            // TODO: gal precomputint pheromone.powf(alpha)? Vis tiek jis keičiasi tik kitoje iteracijoje.
            // Each ant constructs a tour, keep track of the shortest and longest tours
            // found in this iteration.
            let mut iteration_tours = self.construct_ant_tours(&distrib01);
            let stale = iteration_tours.short_tour_length == iteration_tours.long_tour_length;

            // Update pheromone.
            self.update_pheromone(&mut delta_tau_matrix, &iteration_tours);

            // Keep track of the shortest tour.
            if iteration_tours.short_tour_length < self.best_tour.length() {
                self.best_tour = self.ants[iteration_tours.short_tour_ant_idx]
                    .clone_tour(self.tsp_problem.distances());
                dbg!(self.iteration, self.best_tour.length());
            }

            self.iteration += 1;
            self.time += num_cities;

            if self.best_tour.length() == self.tsp_problem.solution_length() {
                println!(
                    "Stopping, reached optimal length in iteration {}",
                    self.iteration
                );
                found_optimal = true;
                break;
            }

            self.exchange_best_tours(&mut cpus_best_tours_buf);
            // We can't use a matrix since the order of buffers from different CPUs is not
            // guranteed.
            let fitness =
                self.calculate_proc_distances_fitness(&cpus_best_tours_buf, &mut proc_distances);
            if self.mpi.is_root {
                dbg!(&proc_distances);
            }

            // TODO: choose exchange partner and perform exchange
            // to do this, we will need to calculate neighbour values for all other CPUs
            let neighbour_values = self.calculate_neighbour_values(&mut proc_distances);
            self.select_exchange_partner(
                &fitness,
                &neighbour_values,
                &mut cpu_random_order,
                &distrib01,
            );

            if stale {
                println!(
                    "Stopping, colony is stale (all ants likely found the same tour). Iteration {}",
                    self.iteration
                );
                break;
            }
            if self.iteration == max_iterations {
                println!("Stopping, reached max iterations count");
                break;
            }

            // Reset ants.
            for a in self.ants.iter_mut() {
                a.reset_to_city(
                    num_cities as u16,
                    CityIndex::new(self.cities_distrib.sample(&mut self.rng)),
                );
            }

            // dbg!(t.elapsed().as_nanos());
        }

        found_optimal
    }

    pub fn number_of_cities(&self) -> u16 {
        self.tsp_problem.number_of_cities()
    }

    pub fn iteration(&self) -> u32 {
        self.iteration
    }

    pub fn best_tour(&self) -> &Tour {
        &self.best_tour
    }

    pub fn set_lowercase_q(&mut self, q: u16) {
        self.lowercase_q = q;
    }

    fn exchange_best_tours(&mut self, recv_buf: &mut [CityIndex]) {
        // self.best_tour.hack_append_length(self.mpi.rank);
        self.best_tour.hack_append_length();
        self.mpi
            .world
            .all_gather_into(self.best_tour.cities(), recv_buf);
        self.best_tour.remove_hack_length();
    }

    fn construct_ant_tours(&mut self, distrib01: &Uniform<Float>) -> ShortLongIterationTours {
        let mut iteration_tours = ShortLongIterationTours {
            short_tour_ant_idx: 0,
            short_tour_length: u32::MAX,
            long_tour_length: 0,
        };
        let num_cities = self.number_of_cities() as usize;

        // Colony is stale if all ants find the same tour. Since actually
        // checking if tours match is expensive, we only check if they are
        // of the same length.
        for (idx, ant) in self.ants.iter_mut().enumerate() {
            for _ in 1..num_cities {
                ant.choose_next_city(self.rng, distrib01, &self.pheromone_matrix, self.alpha);
            }

            let len = ant.tour_length(self.tsp_problem.distances());

            if len < iteration_tours.short_tour_length {
                iteration_tours.short_tour_ant_idx = idx;
                iteration_tours.short_tour_length = len;
            }
            if len > iteration_tours.long_tour_length {
                iteration_tours.long_tour_length = len;
            }
        }

        iteration_tours
    }

    fn update_pheromone(
        &mut self,
        mut delta_tau_matrix: &mut SquareMatrix<Float>,
        iteration_tours: &ShortLongIterationTours,
    ) {
        self.pheromone_matrix.evaporate_pheromone();

        let capital_q = self.capital_q_mul * iteration_tours.long_tour_length as Float;
        // TODO: figure out if this should be calculated using best length so far or only from this iteration.
        let min_tau = capital_q / iteration_tours.short_tour_length as f32;

        for ant in self.ants.iter() {
            ant.update_pheromone(delta_tau_matrix, self.capital_q_mul);
        }

        for x in 0..self.number_of_cities() {
            for y in 0..x {
                let delta = delta_tau_matrix[(x.into(), y.into())];
                self.pheromone_matrix.adjust_pheromone(
                    (CityIndex::new(x), CityIndex::new(y)),
                    if delta > min_tau { delta } else { min_tau },
                );
            }
        }

        delta_tau_matrix.fill(0.0);
    }

    // Since MPI all_gather_into() places buffers from different CPUs in rank order,
    // we don't need to exchange rank info.
    fn calculate_proc_distances_fitness(
        &self,
        cpus_best_tours_buf: &[CityIndex],
        proc_distances: &mut SquareMatrix<u16>,
    ) -> Vec<Float> {
        let num_cities = self.number_of_cities() as usize;
        let mut fitness_scores = Vec::with_capacity(num_cities);
        for (x, best_tour_with_hacks_appended_1) in cpus_best_tours_buf
            .chunks_exact(num_cities + Tour::APPENDED_HACK_ELEMENTS)
            .enumerate()
        {
            fitness_scores
                .push(self.fitness(best_tour_with_hacks_appended_1.get_hack_tour_length()));
            // let cpu1 = CpuInfo {
            // rank: best_tour_with_hacks_appended_1.get_hack_mpi_rank(),
            // best_tour_length: best_tour_with_hacks_appended_1.get_hack_tour_length(),
            // };
            for (y, best_tour_with_hacks_appended_2) in cpus_best_tours_buf
                .chunks_exact(num_cities + Tour::APPENDED_HACK_ELEMENTS)
                .take(x)
                .enumerate()
            {
                // let cpu2 = CpuInfo {
                // rank: best_tour_with_hacks_appended_2.get_hack_mpi_rank(),
                // best_tour_length: best_tour_with_hacks_appended_2.get_hack_tour_length(),
                // };
                let distance = best_tour_with_hacks_appended_1
                    .distance(&best_tour_with_hacks_appended_2[..num_cities]);
                proc_distances[(x, y)] = distance;
                proc_distances[(y, x)] = distance;
                // Distance calculation needs to ignore the appended length and rank.
                // proc_distances.push(CpuDistance {
                // cpu1,
                // cpu2,
                // distance,
                // });
            }
        }

        fitness_scores
    }

    fn fitness(&self, tour_length: u32) -> Float {
        let optimal = self.tsp_problem.solution_length();
        optimal as Float / tour_length as Float
    }

    fn calculate_neighbour_values(&self, proc_distances: &mut SquareMatrix<u16>) -> Vec<Float> {
        let world_size = self.mpi.world_size as usize;
        let mut neighbour_values = Vec::with_capacity(world_size);
        for y in 0..world_size {
            let mut row = proc_distances.row_mut(y);
            row.sort_unstable();

            // First element is going to be 0 since it is distance to self.
            let neighbour: Float =
                row[1..].iter().sum::<u16>() as Float / self.lowercase_q as Float;
            neighbour_values.push(neighbour);
        }

        neighbour_values
    }

    // Returns rank of the chosen CPU.
    fn select_exchange_partner(
        &mut self,
        fitness: &[Float],
        neighbour_values: &[Float],
        cpu_random_order: &mut [usize],
        distrib01: &Uniform<Float>,
    ) -> usize {
        let rank = self.mpi.rank as usize;
        let mut denominator = 0.0;
        let self_neighbour = neighbour_values[rank];
        for i in 0..neighbour_values.len() {
            if i == rank {
                continue;
            }
            denominator += Float::abs(self_neighbour - neighbour_values[i]) * fitness[i];
        }

        cpu_random_order.shuffle(self.rng);
        for i in cpu_random_order.iter().copied() {
            let numerator = Float::abs(self_neighbour - neighbour_values[i]) * fitness[i];
            let p = numerator / denominator;
            if p > distrib01.sample(self.rng) {
                return i;
            }
        }

        // If no CPU is selected, choose a random one.
        cpu_random_order[0]
    }
}

#[derive(Clone, Copy, Debug)]
struct ShortLongIterationTours {
    short_tour_ant_idx: usize,
    short_tour_length: u32,
    long_tour_length: u32,
}

impl ShortLongIterationTours {
    fn is_colony_stale(&self) -> bool {
        self.short_tour_length == self.long_tour_length
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct CpuInfo {
    // rank: i32,
    best_tour_length: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct CpuDistance {
    cpu1: CpuInfo,
    cpu2: CpuInfo,
    distance: u16,
}
