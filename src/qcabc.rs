//! Quick combinatorial artificial bee colony (qCABC).

use std::ops::{Deref, DerefMut};

use mpi::traits::CommunicatorCollectives;
use rand::{
    distributions::{self, Uniform, WeightedIndex},
    prelude::Distribution,
    seq::SliceRandom,
    Rng,
};
use std::cmp::max;

use crate::{
    config::{DistanceT, Float},
    gstm,
    index::CityIndex,
    matrix::{Matrix, SquareMatrix},
    path_usage_matrix::PathUsageMatrix,
    tour::{Tour, TourFunctions},
    tsp_problem::TspProblem,
    utils::{choose_except, Mpi},
};

pub type NeighbourMatrix = Matrix<(CityIndex, DistanceT)>;

pub struct TourExt {
    tour: Tour,
    non_improvement_iters: u32,
    fitness: Float,
    prob_select_by_onlooker: Float,
}

impl TourExt {
    pub fn new(tour: Tour) -> Self {
        let fitness = Self::calc_fitness(&tour);
        Self {
            tour,
            non_improvement_iters: 0,
            fitness,
            prob_select_by_onlooker: 0.0,
        }
    }

    fn calc_fitness(tour: &Tour) -> Float {
        Self::calculate_fitness(tour.length())
    }

    pub fn calculate_fitness(tour_length: DistanceT) -> Float {
        1.0 / (1.0 + tour_length as Float)
    }

    pub fn calc_set_prob_select(&mut self, fit_best: Float) {
        self.prob_select_by_onlooker = (0.9 * self.fitness) / fit_best + 0.1;
    }

    pub fn prob_select_by_onlooker(&self) -> Float {
        debug_assert_ne!(self.prob_select_by_onlooker, 0.0);
        self.prob_select_by_onlooker
    }

    pub fn tour(&self) -> &Tour {
        &self.tour
    }

    pub fn set_tour(&mut self, tour: Tour, path_usage_matrix: &mut PathUsageMatrix) {
        self.fitness = Self::calc_fitness(&tour);
        path_usage_matrix.dec_tour_paths(&self.tour);
        path_usage_matrix.inc_tour_paths(&tour);
        self.tour = tour;
        self.non_improvement_iters = 0;
        // For debug purposes.
        self.prob_select_by_onlooker = 0.0;
    }

    pub fn inc_non_improvement(&mut self) {
        self.non_improvement_iters += 1;
    }

    pub fn non_improvement_iters(&self) -> u32 {
        self.non_improvement_iters
    }

    pub fn fitness(&self) -> Float {
        self.fitness
    }
}

impl Deref for TourExt {
    type Target = Tour;

    fn deref(&self) -> &Self::Target {
        &self.tour
    }
}

pub struct QuickCombArtBeeColony<'a, R: Rng> {
    colony_size: usize,
    iteration: u32, // max iterations will be specified on method evolve_until_optimal
    tours: Vec<TourExt>,
    best_tour: Tour,
    worst_tour_idx: usize,
    worst_tour_length: DistanceT,
    tour_non_improvement_limit: u32,
    tsp_problem: &'a TspProblem,
    // Row i is neighbour list for city CityIndex(i).
    neighbour_lists: NeighbourMatrix,
    path_usage_matrix: PathUsageMatrix, // used for parallelization only
    p_cp: Float,
    p_rc: Float,
    p_l: Float,
    l_min: usize,
    l_max: usize,
    r: Float,
    lowercase_q: usize,
    g: u32,
    k: Float,
    rng: &'a mut R,
    // Distribution for sampling city numbers (suitable for both CityIndex and TourIndex)
    cities_distrib: Uniform<u16>,
    mpi: &'a Mpi<'a>,
}

impl<'a, R: Rng> QuickCombArtBeeColony<'a, R> {
    pub fn new(
        tsp_problem: &'a TspProblem,
        colony_size: usize,
        nl_max: u16,
        capital_l: Float,
        p_cp: Float,
        p_rc: Float,
        p_l: Float,
        l_min: usize,
        l_max: usize,
        r: Float,
        lowercase_q: usize,
        initial_g: u32,
        k: Float,
        rng: &'a mut R,
        mpi: &'a Mpi<'a>,
    ) -> Self {
        let mut tours = Vec::with_capacity(colony_size);
        let number_of_cities = tsp_problem.number_of_cities() as u16;
        let mut path_usage_matrix = PathUsageMatrix::new(number_of_cities);

        let mut best_tour_idx = 0;
        let mut best_tour_length = DistanceT::MAX;
        let mut worst_tour_idx = 0;
        let mut worst_tour_length = 0;
        for i in 0..colony_size {
            let tour = Tour::random(number_of_cities, tsp_problem.distances(), rng);
            if tour.length() < best_tour_length {
                best_tour_length = tour.length();
                best_tour_idx = i;
            } else if tour.length() > worst_tour_length {
                worst_tour_idx = i;
                worst_tour_length - tour.length();
            }
            path_usage_matrix.inc_tour_paths(&tour);
            tours.push(TourExt::new(tour));
        }
        let neighbour_lists = tsp_problem.distances().neighbourhood_lists(nl_max);
        // Eq. 6 in qCABC paper.
        let tour_non_improvement_limit =
            (colony_size as Float * Float::from(number_of_cities)) / capital_l;
        let best_tour = tours[best_tour_idx as usize].tour().clone();

        Self {
            colony_size,
            iteration: 0,
            tours,
            best_tour,
            worst_tour_idx,
            worst_tour_length,
            tour_non_improvement_limit: tour_non_improvement_limit as u32,
            tsp_problem,
            neighbour_lists,
            path_usage_matrix,
            p_cp,
            p_rc,
            p_l,
            l_min,
            l_max,
            r,
            lowercase_q,
            g: initial_g,
            k,
            cities_distrib: Uniform::new(0, number_of_cities).unwrap(),
            rng,
            mpi,
        }
    }

    pub fn colony_size(&self) -> usize {
        self.colony_size
    }

    pub fn iterate_until_optimal(&mut self, max_iterations: u32) -> bool {
        let num_cities = self.number_of_cities() as u16;
        // TODO: what is the split between employed bees and foragers (onlookers)?
        // For now we will assume that all bees are both foragers and onlookers.
        let mut found_optimal = false;
        let distrib_tours = Uniform::new(0, self.colony_size).unwrap();
        let mut tour_distances = SquareMatrix::new(self.tours.len(), 0);
        let distrib01 = distributions::Uniform::new(0.0, 1.0).unwrap();

        let world_size = self.mpi.world_size;
        // We will never exchange with ourselves.
        let other_cpus: Vec<usize> = (0..world_size).filter(|&e| e != self.mpi.rank).collect();
        // Buffer for CPUs' best tours.
        // Tour::APPENDED_HACK_ELEMENTS extra spaces at the end are for tour length.
        let mut cpus_best_tours_buf =
            vec![
                CityIndex::new(0);
                world_size * (self.number_of_cities() + Tour::APPENDED_HACK_ELEMENTS)
            ];
        let mut proc_distances = SquareMatrix::new(world_size, 0);
        // let mut min_delta_tau = config::MIN_DELTA_TAU_INIT;
        let mut last_exchange = self.iteration;

        loop {
            // Employed bees phase.
            self.employed_bees_phase(distrib_tours);

            // Onlooker (forager) bees phase.
            self.onlooker_bees_phase(&mut tour_distances, distrib_tours);

            // Scout phase.
            self.scout_bees_phase();

            self.iteration += 1;

            if self.iteration - last_exchange == self.g {
                self.exchange_best_tours(&mut cpus_best_tours_buf);
                debug_assert_eq!(self.best_tour.number_of_cities(), self.number_of_cities());
                last_exchange = self.iteration;
                let cvg_avg = self.cvg_avg(&cpus_best_tours_buf);
                self.set_exchange_interval(cvg_avg);
                let (_, global_best_tour_length) =
                    self.global_best_tour_length(&cpus_best_tours_buf);
                if global_best_tour_length == self.tsp_problem.solution_length() {
                    found_optimal = true;
                    break;
                }
                // We can't use a matrix since the order of buffers from different CPUs is not
                // guranteed.
                let (fitness, shortest_tour_so_far) =
                    self.calculate_proc_distances(&cpus_best_tours_buf, &mut proc_distances);
                // min_delta_tau = capital_q / shortest_tour_so_far as Float;

                let neighbour_values = self.calculate_neighbour_values(&mut proc_distances);
                let exchange_partner =
                    self.select_exchange_partner(&fitness, &neighbour_values, &other_cpus);
                // if self.mpi.is_root {
                //     dbg!(&proc_distances);
                // }
                let best_partner_tour = &cpus_best_tours_buf[self
                    .hack_tours_buf_idx(exchange_partner)
                    ..self.hack_tours_buf_idx(exchange_partner + 1)];
                self.replace_worst_self_tour_with_best_partner_tour(
                    &best_partner_tour[..self.tsp_problem.number_of_cities()],
                    best_partner_tour.get_hack_tour_length(),
                );

                if self.mpi.is_root {
                    eprintln!(
                        "Done an exchange on iteration {}, global best tour length {}, next exchange in {} iterations, cvg_avg {}",
                        self.iteration, global_best_tour_length, self.g, cvg_avg
                    );
                }
            } else if self.best_tour.length() == self.tsp_problem.solution_length() {
                eprintln!(
                    "Stopping, reached optimal length in iteration {}",
                    self.iteration
                );
                found_optimal = true;
                break;
            }

            if self.iteration == max_iterations {
                self.exchange_best_tours(&mut cpus_best_tours_buf);
                let (idx, global_best_tour_length) =
                    self.global_best_tour_length(&cpus_best_tours_buf);
                self.best_tour = Tour::clone_from_cities(
                    &cpus_best_tours_buf[self.hack_tours_buf_idx(idx)
                        ..(self.hack_tours_buf_idx(idx + 1) - Tour::APPENDED_HACK_ELEMENTS)],
                    global_best_tour_length,
                    self.tsp_problem.distances(),
                );
                if self.mpi.is_root {
                    eprintln!("Stopping, reached max iterations count");
                }
                break;
            }
        }

        found_optimal
    }

    fn hack_tours_buf_idx(&self, idx: usize) -> usize {
        (self.number_of_cities() + Tour::APPENDED_HACK_ELEMENTS) * idx
    }

    pub fn number_of_cities(&self) -> usize {
        self.tsp_problem.number_of_cities()
    }

    /// Mean distance of x_i to all the other tours.
    // TODO: qCABC paper 9 formulėje klaida, ten reikia, kad m != i (nors tai nieko
    // nekeičia, atstumas iki savęs = 0).
    fn md_i(&self, i: usize, tour_distances: &SquareMatrix<u16>) -> Float {
        let sum_d: u32 = tour_distances.row(i).iter().copied().map(u32::from).sum();

        sum_d as Float / (self.tours.len() - 1) as Float
    }

    fn fill_tour_distance_matrix(&self, matrix: &mut SquareMatrix<u16>) {
        for (x, tour1) in self.tours.iter().enumerate() {
            for (y, tour2) in self.tours.iter().enumerate().take(x) {
                let dist = tour1.distance(tour2) as u16;
                matrix[(x, y)] = dist;
                matrix[(y, x)] = dist;
            }
        }
    }

    fn employed_bees_phase(&mut self, distrib_tours: Uniform<usize>) {
        for idx in 0..self.colony_size {
            let v_i = gstm::generate_neighbour(
                &self.tours[idx],
                &self.tours[choose_except(distrib_tours, idx, self.rng)],
                self.p_rc,
                self.p_cp,
                self.p_l,
                self.l_min,
                self.l_max,
                &self.neighbour_lists,
                self.tsp_problem.distances(),
                self.cities_distrib,
                self.rng,
            );
            if v_i.is_shorter_than(&self.tours[idx]) {
                self.tours[idx].set_tour(v_i, &mut self.path_usage_matrix);
            }
        }

        // Calculate the probabilities of being selected by onlookers.
        let fit_best = self
            .tours
            .iter()
            .map(|t| (t.length(), t.fitness()))
            .min_by_key(|&(len, fit)| len)
            .unwrap()
            .1;
        for t in self.tours.iter_mut() {
            t.calc_set_prob_select(fit_best);
        }
    }

    fn onlooker_bees_phase(
        &mut self,
        tour_distances: &mut SquareMatrix<u16>,
        distrib_tours: Uniform<usize>,
    ) {
        let distrib_weighted =
            WeightedIndex::new(self.tours.iter().map(TourExt::prob_select_by_onlooker)).unwrap();
        self.fill_tour_distance_matrix(tour_distances);

        for _ in 0..self.colony_size {
            let t_idx = distrib_weighted.sample(self.rng);
            let threshold = self.r * self.md_i(t_idx, tour_distances);

            let (best_neighbour_idx, best_neighbour_length) = tour_distances
                .row(t_idx)
                .iter()
                .copied()
                .enumerate()
                .filter_map(|(i, dist)| {
                    (Float::from(dist) <= threshold).then(|| (i, self.tours[i].length()))
                })
                .min_by_key(|&(i, length)| length)
                .unwrap();

            let t = &self.tours[best_neighbour_idx];
            let v_i = gstm::generate_neighbour(
                t,
                &self.tours[choose_except(distrib_tours, best_neighbour_idx, self.rng)],
                self.p_rc,
                self.p_cp,
                self.p_l,
                self.l_min,
                self.l_max,
                &self.neighbour_lists,
                self.tsp_problem.distances(),
                self.cities_distrib,
                self.rng,
            );
            if v_i.length() < best_neighbour_length {
                self.tours[best_neighbour_idx].set_tour(v_i, &mut self.path_usage_matrix);
            }
        }
    }

    fn scout_bees_phase(&mut self) {
        let num_cities = self.tsp_problem.number_of_cities() as u16;
        // Also, update the best tour found so far.
        let (mut best_tour_idx, mut best_tour_length) = (0, DistanceT::MAX);
        let (mut worst_tour_idx, mut worst_tour_length) = (0, 0);
        for (idx, tour) in self.tours.iter_mut().enumerate() {
            if tour.non_improvement_iters() == self.tour_non_improvement_limit {
                tour.set_tour(
                    Tour::random(num_cities, self.tsp_problem.distances(), self.rng),
                    &mut self.path_usage_matrix,
                );
            }
            if tour.length() < best_tour_length {
                best_tour_length = tour.length();
                best_tour_idx = idx;
            } else if tour.length() > worst_tour_length {
                worst_tour_length = tour.length();
                worst_tour_idx = idx;
            }
        }
        // New best could be longer, because if the best tour is not improved
        // for some generations, it is replaced.
        if best_tour_length < self.best_tour.length() {
            self.best_tour = self.tours[best_tour_idx].tour().clone();
        }
        self.worst_tour_idx = worst_tour_idx;
        self.worst_tour_length = worst_tour_length;
    }

    fn exchange_best_tours(&mut self, recv_buf: &mut [CityIndex]) {
        let cvg = self.path_usage_matrix.convergence();
        // self.best_tour.hack_append_length(self.mpi.rank);
        self.best_tour.hack_append_length_cvg(cvg);
        self.mpi
            .world
            .all_gather_into(self.best_tour.cities(), recv_buf);
        self.best_tour.remove_hack_length_cvg();
    }

    /// Returns best tour index and its length.
    fn global_best_tour_length(&self, cpus_best_tours_buf: &[CityIndex]) -> (usize, DistanceT) {
        let chunk_size = self.number_of_cities() + Tour::APPENDED_HACK_ELEMENTS;

        cpus_best_tours_buf
            .chunks_exact(chunk_size)
            .map(TourFunctions::get_hack_tour_length)
            .enumerate()
            .fold(
                (0, DistanceT::MAX),
                |(best_tour_idx, best_tour_length), (idx, tour_with_hacks_len)| {
                    if tour_with_hacks_len < best_tour_length {
                        (idx, tour_with_hacks_len)
                    } else {
                        (best_tour_idx, best_tour_length)
                    }
                },
            )
    }

    fn set_exchange_interval(&mut self, cvg_avg: Float) {
        if cvg_avg >= 0.8 || cvg_avg <= 0.2 {
            let new_g = self.g as Float + ((0.5 - cvg_avg) * self.k as Float);
            debug_assert!(new_g >= 0.0);
            self.g = max(new_g as u32, 1);
        }
    }

    fn cvg_avg(&self, cpus_best_tours_buf: &[CityIndex]) -> Float {
        let chunk_size = self.number_of_cities() + Tour::APPENDED_HACK_ELEMENTS;
        debug_assert_eq!(cpus_best_tours_buf.len() % chunk_size, 0);
        debug_assert_eq!(cpus_best_tours_buf.len() / chunk_size, self.mpi.world_size);

        let cvg_sum: Float = cpus_best_tours_buf
            .chunks_exact(chunk_size)
            .map(TourFunctions::get_hack_cvg)
            .sum();

        cvg_sum / self.mpi.world_size as Float
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

    // Since MPI all_gather_into() places buffers from different CPUs in rank order,
    // we don't need to exchange rank info.
    // Returns distance values and shortes global tour so far.
    fn calculate_proc_distances(
        &self,
        cpus_best_tours_buf: &[CityIndex],
        proc_distances: &mut SquareMatrix<u16>,
    ) -> (Vec<Float>, DistanceT) {
        let mut shortest_tour_so_far = DistanceT::MAX;
        let num_cities = self.number_of_cities();
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

            // Since we are not setting distance to self (pointless), it gets corrupted
            // when we sort the rows of the distance table.
            for (y, best_tour_with_hacks_appended_2) in cpus_best_tours_buf
                .chunks_exact(chunk_size)
                .take(x)
                .enumerate()
            {
                let distance = best_tour_with_hacks_appended_1[..num_cities]
                    .distance(&best_tour_with_hacks_appended_2[..num_cities]);
                proc_distances[(x, y)] = distance as u16;
                proc_distances[(y, x)] = distance as u16;
            }
        }

        (fitness_scores, shortest_tour_so_far)
    }

    fn fitness(&self, tour_length: DistanceT) -> Float {
        // self.tsp_problem.solution_length() as Float / tour_length as Float
        TourExt::calculate_fitness(tour_length)
    }

    fn calculate_neighbour_values(&self, proc_distances: &mut SquareMatrix<u16>) -> Vec<Float> {
        let mut neighbour_values = Vec::with_capacity(self.mpi.world_size);
        for y in 0..self.mpi.world_size {
            let mut row = proc_distances.row_mut(y);
            row.sort_unstable();

            // First element is going to be 0 since it is distance to self.
            let neighbour = Float::from(row[1..].iter().take(self.lowercase_q).sum::<u16>())
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
        other_cpus: &[usize],
    ) -> usize {
        let mut denominator = 0.0;
        let self_neighbour = neighbour_values[self.mpi.rank];
        for i in 0..neighbour_values.len() {
            if i == self.mpi.rank {
                continue;
            }
            denominator += Float::abs(self_neighbour - neighbour_values[i]) * fitness[i];
        }

        *other_cpus
            .choose_weighted(self.rng, |&cpu| {
                (Float::abs(self_neighbour - neighbour_values[cpu]) * fitness[cpu]) / denominator
            })
            .unwrap()
    }

    fn replace_worst_self_tour_with_best_partner_tour(
        &mut self,
        partner_tour: &[CityIndex],
        partner_tour_length: DistanceT,
    ) {
        self.tours[self.worst_tour_idx].set_tour(
            Tour::clone_from_cities(
                partner_tour,
                partner_tour_length,
                self.tsp_problem.distances(),
            ),
            &mut self.path_usage_matrix,
        );
    }
}
