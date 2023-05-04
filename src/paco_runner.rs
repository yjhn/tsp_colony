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
    config::{self, Float},
    matrix::SquareMatrix,
    pheromone_visibility_matrix::PheromoneVisibilityMatrix,
    tour::{CityIndex, Tour, TourFunctions},
    tsp_problem::TspProblem,
    utils::{maxf, Mpi},
};

/// Runs the PACO algorithm as defined in paper:
/// "A parallel ant colony algorithm on massively parallel processors and
/// its convergence analysis for the travelling salesman problem"
pub struct PacoRunner<'a, R: Rng + SeedableRng> {
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
    lowercase_q: usize,
    TODO: exchange generations always repeat with the same values: 6, 12, 18, 24, ... (6 is added after every exchange for some reason)
    g: u32,
    k: u32,
    mpi: &'a Mpi<'a>,
    cities_distrib: Uniform<u16>,
}

impl<'a, R: Rng + SeedableRng> PacoRunner<'a, R> {
    pub fn new(
        ant_count: usize,
        mut rng: &'a mut R,
        tsp_problem: &'a TspProblem,
        initial_trail_intensity: Float,
        alpha: Float,
        beta: Float,
        capital_q_mul: Float,
        ro: Float,
        lowercase_q: usize,
        init_g: u32,
        k: u32,
        mpi: &'a Mpi,
    ) -> PacoRunner<'a, R> {
        let city_count = tsp_problem.number_of_cities() as u16;
        let pheromone_matrix = PheromoneVisibilityMatrix::new(
            tsp_problem.number_of_cities(),
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

        PacoRunner {
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
            g: init_g,
            k,
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

    pub fn set_g(&mut self, g: u32) {
        self.g = g;
    }

    pub fn set_k(&mut self, k: u32) {
        self.k = k;
    }

    /// Resets pheromone intensity to `intensity`.
    pub fn reset_pheromone(&mut self, intensity: Float) {
        self.pheromone_matrix.reset_pheromone(intensity);
    }

    pub fn reset_all_state(&mut self, init_g: u32) {
        // Reset ants.
        let num_cities = self.number_of_cities() as u16;
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
        self.g = init_g;
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
        // let mut min_delta_tau = config::MIN_DELTA_TAU_INIT;
        let mut last_exchange = self.iteration;

        loop {
            // TODO: gal precomputint pheromone.powf(alpha)? Vis tiek jis keičiasi tik kitoje iteracijoje.
            // Each ant constructs a tour, keep track of the shortest and longest tours
            // found in this iteration.
            let iteration_tours = self.construct_ant_tours(&distrib01);

            // Keep track of the shortest tour.
            if iteration_tours.short_tour_length < self.best_tour.length() {
                self.best_tour = self.ants[iteration_tours.short_tour_ant_idx]
                    .clone_tour(self.tsp_problem.distances());
                // dbg!(self.iteration, self.best_tour.length());
            }

            self.iteration += 1;
            self.time += num_cities;

            let capital_q = iteration_tours.long_tour_length as Float * self.capital_q_mul;
            let min_delta_tau = capital_q / iteration_tours.short_tour_length as Float;
            // update pheromone.
            self.update_pheromone(
                &mut delta_tau_matrix,
                &iteration_tours,
                capital_q,
                min_delta_tau,
            );
            //TODO: exchange at time interval, not at every iteration.
            if self.iteration - last_exchange == self.g {
                self.exchange_best_tours(&mut cpus_best_tours_buf);
                last_exchange = self.iteration;
                let cvg_avg = self.cvg_avg(&cpus_best_tours_buf);
                self.set_exchange_interval(cvg_avg);
                let global_best_tour_length = self.global_best_tour_length(&cpus_best_tours_buf);
                if global_best_tour_length == self.tsp_problem.solution_length() {
                    found_optimal = true;
                    break;
                }
                // We can't use a matrix since the order of buffers from different CPUs is not
                // guranteed.
                let (fitness, shortest_tour_so_far) = self
                    .calculate_proc_distances_fitness(&cpus_best_tours_buf, &mut proc_distances);
                // min_delta_tau = capital_q / shortest_tour_so_far as Float;

                let neighbour_values = self.calculate_neighbour_values(&mut proc_distances);
                let exchange_partner = self.select_exchange_partner(
                    &fitness,
                    &neighbour_values,
                    &mut cpu_random_order,
                    &distrib01,
                );
                // if self.mpi.is_root {
                //     dbg!(&proc_distances);
                // }
                let best_partner_tour = &cpus_best_tours_buf[(exchange_partner
                    * (num_cities + Tour::APPENDED_HACK_ELEMENTS))
                    ..((exchange_partner + 1) * (num_cities + Tour::APPENDED_HACK_ELEMENTS))];
                self.update_pheromones_from_partner(
                    &best_partner_tour[..best_partner_tour.len() - Tour::APPENDED_HACK_ELEMENTS],
                    fitness[exchange_partner],
                    fitness[self.mpi.rank as usize],
                    best_partner_tour.get_hack_tour_length(),
                    &mut delta_tau_matrix,
                    min_delta_tau,
                    capital_q,
                );

                if self.mpi.is_root {
                    eprintln!(
                        "Done an exchange on iteration {}, global best tour length {}, next exchange in {} iterations",
                        self.iteration, global_best_tour_length, self.g
                    );
                }
            } else if self.best_tour.length() == self.tsp_problem.solution_length() {
                println!(
                    "Stopping, reached optimal length in iteration {}",
                    self.iteration
                );
                found_optimal = true;
                break;
            }

            if iteration_tours.is_colony_stale() {
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

    pub fn number_of_cities(&self) -> usize {
        self.tsp_problem.number_of_cities()
    }

    pub fn iteration(&self) -> u32 {
        self.iteration
    }

    pub fn best_tour(&self) -> &Tour {
        &self.best_tour
    }

    pub fn set_lowercase_q(&mut self, q: usize) {
        self.lowercase_q = q;
    }

    fn exchange_best_tours(&mut self, recv_buf: &mut [CityIndex]) {
        let cvg = self.calculate_cvg();
        // self.best_tour.hack_append_length(self.mpi.rank);
        self.best_tour.hack_append_length_cvg(cvg);
        self.mpi
            .world
            .all_gather_into(self.best_tour.cities(), recv_buf);
        self.best_tour.remove_hack_length_cvg();
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
        capital_q: Float,
        min_delta_tau: Float,
    ) -> Float {
        self.pheromone_matrix.evaporate_pheromone();
        // let capital_q = self.capital_q_mul * iteration_tours.long_tour_length as Float;
        // TODO: figure out if this should be calculated using best length so far or only from this iteration.
        // let min_delta_tau = capital_q / iteration_tours.short_tour_length as f32;

        for ant in self.ants.iter() {
            ant.update_pheromone(delta_tau_matrix, capital_q);
        }

        for x in (0..self.number_of_cities() as u16) {
            for y in 0..x {
                let delta = delta_tau_matrix[(x.into(), y.into())];
                self.pheromone_matrix.adjust_pheromone(
                    (CityIndex::new(x), CityIndex::new(y)),
                    maxf(delta, min_delta_tau),
                );
            }
        }

        delta_tau_matrix.fill(0.0);
        min_delta_tau
    }

    // Since MPI all_gather_into() places buffers from different CPUs in rank order,
    // we don't need to exchange rank info.
    // Returns ditness values and shortes global tour so far.
    fn calculate_proc_distances_fitness(
        &self,
        cpus_best_tours_buf: &[CityIndex],
        proc_distances: &mut SquareMatrix<u16>,
    ) -> (Vec<Float>, u32) {
        let mut shortest_tour_so_far = u32::MAX;
        let num_cities = self.number_of_cities() as usize;
        let chunk_size = num_cities + Tour::APPENDED_HACK_ELEMENTS;
        debug_assert_eq!(cpus_best_tours_buf.len() % chunk_size, 0);
        let mut fitness_scores = Vec::with_capacity(num_cities);
        for (x, best_tour_with_hacks_appended_1) in
            cpus_best_tours_buf.chunks_exact(chunk_size).enumerate()
        {
            let len = best_tour_with_hacks_appended_1.get_hack_tour_length();
            if len < shortest_tour_so_far {
                shortest_tour_so_far = len;
            }
            fitness_scores.push(self.fitness(len));
            // let cpu1 = CpuInfo {
            // rank: best_tour_with_hacks_appended_1.get_hack_mpi_rank(),
            // best_tour_length: best_tour_with_hacks_appended_1.get_hack_tour_length(),
            // };

            // Since we are not setting distance to self (pointless), it gets corrupted
            // when we sort the rows of the distance table.
            for (y, best_tour_with_hacks_appended_2) in cpus_best_tours_buf
                .chunks_exact(chunk_size)
                .take(x)
                .enumerate()
            {
                // let cpu2 = CpuInfo {
                // rank: best_tour_with_hacks_appended_2.get_hack_mpi_rank(),
                // best_tour_length: best_tour_with_hacks_appended_2.get_hack_tour_length(),
                // };
                let distance = best_tour_with_hacks_appended_1[..num_cities]
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

        (fitness_scores, shortest_tour_so_far)
    }

    // TODO: what fitness function to use is not specified in the paper, maybe it does not matter?
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
            let neighbour: Float = row[1..].iter().take(self.lowercase_q).sum::<u16>() as Float
                / self.lowercase_q as Float;
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

    // TODO: either regular update OR this update can occur, not both after one iteration.
    fn update_pheromones_from_partner(
        &mut self,
        best_partner_tour: &[CityIndex],
        partner_fitness: Float,
        self_fitness: Float,
        best_partner_tour_length: u32,
        mut delta_tau_matrix: &mut SquareMatrix<Float>,
        delta_tau_min: Float,
        capital_q: Float,
    ) {
        // evaporation is probably unneccessary here?
        self.pheromone_matrix.evaporate_pheromone();
        // Formula 7 in the PACO paper.
        let fit_gh = self_fitness / (self_fitness + partner_fitness);
        let delta_tau_g = capital_q / self.best_tour.length() as Float;
        let delta_g = maxf(delta_tau_g, delta_tau_min);
        let delta_tau_h = capital_q / best_partner_tour_length as Float;
        let delta_h = maxf(delta_tau_h, delta_tau_min);

        // TODO: ar delta_tau_min skaičiuojamas kiekvienam procesoriui atskirai, ar visiems bendrai?
        // kolkas naudoju lokaliai suskaičiuotą
        // TODO: taip pat gauti delta_tau ir delta_tau_min iš kitų procesorių (vėlgi pridėti prie Tour galo)
        // let delta_tau = fit_gh * delta_tau_g + fit_gh * delta_tau_h;
        let delta_tau_final_g = fit_gh * delta_tau_g + fit_gh * delta_tau_min;
        let delta_tau_final_h = fit_gh * delta_tau_min + fit_gh * delta_tau_h;
        // update delta tau matrix
        // self best path
        self.best_tour
            .cities()
            .update_pheromone(delta_tau_matrix, delta_tau_final_g);
        // partner best path
        best_partner_tour.update_pheromone(delta_tau_matrix, delta_tau_final_h);
        for x in (0..self.number_of_cities() as u16) {
            for y in 0..x {
                let delta = delta_tau_matrix[(x.into(), y.into())];
                self.pheromone_matrix.adjust_pheromone(
                    (CityIndex::new(x), CityIndex::new(y)),
                    maxf(delta, delta_tau_min),
                );
            }
        }

        delta_tau_matrix.fill(0.0);
    }

    fn calculate_cvg(&self) -> Float {
        let (max_pher, avg_pher) = self.pheromone_matrix.max_avg_pheromone();

        let first = 1.0
            / (self.pheromone_matrix.pheromone_element_count() as Float * (max_pher - avg_pher));

        let sum_diff_pher = self.pheromone_matrix.sum_diff_pher(avg_pher);

        first * sum_diff_pher
    }

    fn cvg_avg(&self, cpus_best_tours_buf: &[CityIndex]) -> Float {
        let chunk_size = self.number_of_cities() + Tour::APPENDED_HACK_ELEMENTS;
        debug_assert_eq!(cpus_best_tours_buf.len() % chunk_size, 0);

        let cvg_sum: Float = cpus_best_tours_buf
            .chunks_exact(chunk_size)
            .map(|tour_with_hacks| tour_with_hacks.get_hack_cvg())
            .sum();

        cvg_sum / self.mpi.world_size as Float
    }

    fn global_best_tour_length(&self, cpus_best_tours_buf: &[CityIndex]) -> u32 {
        let chunk_size = self.number_of_cities() + Tour::APPENDED_HACK_ELEMENTS;

        cpus_best_tours_buf
            .chunks_exact(chunk_size)
            .map(|tour_with_hacks| tour_with_hacks.get_hack_tour_length())
            .max()
            .unwrap()
    }

    fn set_exchange_interval(&mut self, cvg_avg: Float) {
        if cvg_avg >= 0.8 || cvg_avg <= 0.2 {
            let new_g = self.g as Float + (0.5 - cvg_avg) * self.k as Float;
            debug_assert!(new_g >= 0.0);
            self.g = max(new_g as u32, 1);
        }
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