//! Quick combinatorial artificial bee colony (qCABC).

use std::{
    ops::{Deref, DerefMut},
    time::Instant,
};

use mpi::traits::CommunicatorCollectives;
use rand::{
    distributions::{self, Uniform, WeightedIndex},
    prelude::Distribution,
    seq::SliceRandom,
    Rng,
};
use std::cmp::max;

use crate::{
    arguments::Algorithm,
    config::{DistanceT, Float},
    gstm,
    index::CityIndex,
    matrix::{Matrix, SquareMatrix},
    path_usage_matrix::PathUsageMatrix,
    tour::{Tour, TourFunctions},
    tsp_problem::TspProblem,
    utils::{avg, IterateResult, Mpi},
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
        // path_usage_matrix.dec_tour_paths(&self.tour);
        // path_usage_matrix.inc_tour_paths(&tour);
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
    algo: Algorithm,
    colony_size: u32,
    iteration: u32, // max iterations will be specified on method evolve_until_optimal
    tours: Vec<TourExt>,
    best_tour: Tour,
    global_best_tour_length: DistanceT,
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
    // Distribution for range [0; 1)
    distrib01: Uniform<Float>,
    mpi: &'a Mpi<'a>,
}

impl<'a, R: Rng> QuickCombArtBeeColony<'a, R> {
    pub fn new(
        algo: Algorithm,
        tsp_problem: &'a TspProblem,
        colony_size: u32,
        nl_max: u16, // neighbours in neighbour list
        capital_l: Float,
        p_cp: Float,
        p_rc: Float,
        p_l: Float,
        l_min: usize,
        l_max_mul: Float,
        r: Float,
        lowercase_q: usize,
        initial_g: u32,
        k: Float,
        rng: &'a mut R,
        mpi: &'a Mpi<'a>,
    ) -> Self {
        let mut tours = Vec::with_capacity(colony_size as usize);
        let number_of_cities = tsp_problem.number_of_cities() as u16;
        let cities_distrib = Uniform::new(0, number_of_cities).unwrap();
        let mut path_usage_matrix = PathUsageMatrix::new(
            number_of_cities,
            Float::from(number_of_cities) / colony_size as Float,
        );

        let mut best_tour_idx = 0;
        let mut best_tour_length = DistanceT::MAX;
        let mut worst_tour_idx = 0;
        let mut worst_tour_length = 0;
        for i in 0..colony_size {
            let tour = Tour::nearest_neighbour(
                number_of_cities,
                tsp_problem.distances(),
                cities_distrib,
                rng,
            );
            if tour.length() < best_tour_length {
                best_tour_idx = i as usize;
                best_tour_length = tour.length();
            } else if tour.length() > worst_tour_length {
                worst_tour_idx = i as usize;
                worst_tour_length = tour.length();
            }
            // path_usage_matrix.inc_tour_paths(&tour);
            tours.push(TourExt::new(tour));
        }
        let neighbour_lists = tsp_problem.distances().neighbourhood_lists(nl_max);
        // Eq. 6 in qCABC paper.
        let tour_non_improvement_limit =
            (colony_size as Float * Float::from(number_of_cities)) / capital_l;
        let best_tour = tours[best_tour_idx].tour().clone();

        Self {
            algo,
            colony_size,
            iteration: 0,
            tours,
            best_tour,
            global_best_tour_length: DistanceT::MAX,
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
            l_max: (Float::from(number_of_cities) * l_max_mul) as usize,
            r,
            lowercase_q,
            g: initial_g,
            k,
            cities_distrib,
            distrib01: Uniform::new(0.0, 1.0).unwrap(),
            rng,
            mpi,
        }
    }

    pub fn iterate_until_optimal(&mut self, max_iterations: u32) -> IterateResult {
        // TODO: what is the split between employed bees and foragers (onlookers)?
        // For now we will assume that all bees are both foragers and onlookers.
        let mut found_optimal_tour = false;
        let distrib_tours = Uniform::new(0, self.colony_size as usize).unwrap();
        let mut tour_distances = SquareMatrix::new(self.tours.len(), 0);

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
        let mut shortest_iteration_tours = Vec::with_capacity((max_iterations / 4) as usize);
        let mut timing_info = Vec::with_capacity(max_iterations as usize);

        loop {
            let it_start = Instant::now();
            let mut is_exchange_iter = false;
            // Iterations are numbered from 1.
            self.iteration += 1;

            // Employed bees phase.
            self.employed_bees_phase(distrib_tours);

            // Onlooker (forager) bees phase.
            if self.algo == Algorithm::Qcabc {
                self.onlooker_bees_phase_qcabc(&mut tour_distances, distrib_tours);
            } else {
                self.onlooker_bees_phase_cabc(distrib_tours);
            }
            // Scout phase.
            self.scout_bees_phase(&mut shortest_iteration_tours);

            if self.mpi.world_size > 1 {
                if self.iteration - last_exchange == self.g {
                    is_exchange_iter = true;
                    self.exchange_best_tours(&mut cpus_best_tours_buf);
                    last_exchange = self.iteration;
                    // let cvg_avg = self.cvg_avg(&cpus_best_tours_buf);
                    // self.set_exchange_interval(cvg_avg);
                    let (fitness, global_best_tour_length) =
                        self.calculate_proc_distances(&cpus_best_tours_buf, &mut proc_distances);
                    if global_best_tour_length < self.global_best_tour_length {
                        self.global_best_tour_length = global_best_tour_length;
                        self.update_shortest_iteration_tours(
                            global_best_tour_length,
                            &mut shortest_iteration_tours,
                        );
                    }
                    if self.global_best_tour_length == self.tsp_problem.solution_length() {
                        eprintln!("Globally found best tour in iteration {}", self.iteration);
                        found_optimal_tour = true;
                        break;
                    }

                    let neighbour_values = self.calculate_neighbour_values(&mut proc_distances);
                    let exchange_partner =
                        self.select_exchange_partner(&fitness, &neighbour_values, &other_cpus);
                    let best_partner_tour = &cpus_best_tours_buf[self
                        .hack_tours_buf_idx(exchange_partner)
                        ..self.hack_tours_buf_idx(exchange_partner + 1)];
                    self.replace_worst_self_tour_with_best_partner_tour(
                        best_partner_tour.without_hacks(),
                        best_partner_tour.get_hack_tour_length(),
                    );

                    // if self.mpi.is_root {
                    // eprintln!(
                    // "Done an exchange on iteration {}, global best tour length {}, next exchange in {} iterations, cvg_avg {}",
                    // self.iteration, global_best_tour_length, self.g, cvg_avg
                    // );
                    // }
                }
            } else if self.best_tour.length() == self.tsp_problem.solution_length() {
                eprintln!(
                    "Stopping, found optimal tour in iteration {}",
                    self.iteration
                );
                found_optimal_tour = true;
                break;
            }

            if self.iteration == max_iterations {
                self.exchange_best_tours(&mut cpus_best_tours_buf);
                let (idx, global_best_tour_length) =
                    self.global_best_tour_length(&cpus_best_tours_buf);
                self.global_best_tour_length = global_best_tour_length;
                self.update_shortest_iteration_tours(
                    global_best_tour_length,
                    &mut shortest_iteration_tours,
                );
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

            timing_info.push((it_start.elapsed().as_micros(), is_exchange_iter));
        }

        let iter_time_non_exch = timing_info
            .iter()
            .copied()
            .filter_map(|e| (!e.1).then_some(e.0));
        let avg_iter_time_non_exch = avg(iter_time_non_exch);
        let iter_time_exch = timing_info
            .iter()
            .copied()
            .filter_map(|e| (e.1).then_some(e.0));
        let avg_iter_time_exch = avg(iter_time_exch);

        IterateResult {
            found_optimal_tour,
            shortest_iteration_tours,
            avg_iter_time_non_exch_micros: avg_iter_time_non_exch,
            avg_iter_time_exch_micros: avg_iter_time_exch,
        }
    }

    fn update_shortest_iteration_tours(
        &self,
        length: DistanceT,
        shortest_iteration_tours: &mut Vec<(u32, DistanceT)>,
    ) {
        if let Some((it, len)) = shortest_iteration_tours.last_mut() {
            if *it == self.iteration {
                *len = length;
                return;
            }
        }
        shortest_iteration_tours.push((self.iteration, length));
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

    // TODO: maybe each tour can have a flag has_changed that is set if the tour has been
    // changed in this iteration. If the flag is not set for both, do not recalculate distance.
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
            // Increase tour non-improvement iterations.
            self.tours[idx as usize].inc_non_improvement();
            let (t_i, t_k) =
                Self::choose_except(&self.tours, distrib_tours, idx as usize, self.rng);
            let v_i = gstm::generate_neighbour(
                t_i,
                t_k,
                self.p_rc,
                self.p_cp,
                self.p_l,
                self.l_min,
                self.l_max,
                &self.neighbour_lists,
                self.tsp_problem.distances(),
                self.cities_distrib,
                self.distrib01,
                self.rng,
            );
            if v_i.is_shorter_than(&self.tours[idx as usize]) {
                self.tours[idx as usize].set_tour(v_i, &mut self.path_usage_matrix);
            }
        }

        // Calculate the probabilities of being selected by onlookers.
        let fit_best = self
            .tours
            .iter()
            // Floats do not implement Ord, so we compare tours by the length.
            .map(|t| (t.length(), t.fitness()))
            .min_by_key(|&(len, _fit)| len)
            .unwrap()
            .1;
        for t in self.tours.iter_mut() {
            t.calc_set_prob_select(fit_best);
        }
    }

    fn onlooker_bees_phase_qcabc(
        &mut self,
        tour_distances: &mut SquareMatrix<u16>,
        distrib_tours: Uniform<usize>,
    ) {
        let distrib_weighted =
            WeightedIndex::new(self.tours.iter().map(TourExt::prob_select_by_onlooker)).unwrap();
        self.fill_tour_distance_matrix(tour_distances);
        // dbg!(self.mpi.rank, self.iteration);

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
                .min_by_key(|&(_i, length)| length)
                .unwrap();

            let (t_i, t_k) =
                Self::choose_except(&self.tours, distrib_tours, best_neighbour_idx, self.rng);
            let v_i = gstm::generate_neighbour(
                t_i,
                t_k,
                self.p_rc,
                self.p_cp,
                self.p_l,
                self.l_min,
                self.l_max,
                &self.neighbour_lists,
                self.tsp_problem.distances(),
                self.cities_distrib,
                self.distrib01,
                self.rng,
            );
            if v_i.length() < best_neighbour_length {
                self.tours[best_neighbour_idx].set_tour(v_i, &mut self.path_usage_matrix);
            }
        }
        // dbg!(self.mpi.rank, self.iteration);
    }

    fn onlooker_bees_phase_cabc(&mut self, distrib_tours: Uniform<usize>) {
        let distrib_weighted =
            WeightedIndex::new(self.tours.iter().map(TourExt::prob_select_by_onlooker)).unwrap();
        for _ in 0..self.colony_size {
            let t_idx = distrib_weighted.sample(self.rng);

            let (t_i, t_k) = Self::choose_except(&self.tours, distrib_tours, t_idx, self.rng);
            let v_i = gstm::generate_neighbour(
                t_i,
                t_k,
                self.p_rc,
                self.p_cp,
                self.p_l,
                self.l_min,
                self.l_max,
                &self.neighbour_lists,
                self.tsp_problem.distances(),
                self.cities_distrib,
                self.distrib01,
                self.rng,
            );
            if v_i.length() < t_i.length() {
                self.tours[t_idx].set_tour(v_i, &mut self.path_usage_matrix);
            }
        }
        // dbg!(self.mpi.rank, self.iteration);
    }

    fn scout_bees_phase(&mut self, shortest_iteration_tours: &mut Vec<(u32, DistanceT)>) {
        let num_cities = self.tsp_problem.number_of_cities() as u16;
        // Also, update the best and worst tours found so far.
        let (mut best_tour_idx, mut best_tour_length) = (0, DistanceT::MAX);
        let (mut worst_tour_idx, mut worst_tour_length) = (0, 0);
        for (idx, tour) in self.tours.iter_mut().enumerate() {
            if tour.non_improvement_iters() == self.tour_non_improvement_limit {
                tour.set_tour(
                    Tour::nearest_neighbour(
                        num_cities,
                        self.tsp_problem.distances(),
                        self.cities_distrib,
                        self.rng,
                    ),
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
            if self.mpi.is_root {
                eprintln!(
                    "New best tour in iteration {}, length {}",
                    self.iteration, best_tour_length
                );
            }
        }
        if best_tour_length < self.global_best_tour_length {
            if self.mpi.is_root {
                shortest_iteration_tours.push((self.iteration, self.best_tour.length()));
            }
            self.global_best_tour_length = best_tour_length
        }
        self.worst_tour_idx = worst_tour_idx;
        self.worst_tour_length = worst_tour_length;
    }

    fn exchange_best_tours(&mut self, recv_buf: &mut [CityIndex]) {
        // let cvg = self.path_usage_matrix.convergence();
        // dbg!(cvg);
        // self.best_tour.hack_append_length(self.mpi.rank);
        self.best_tour.hack_append_length_cvg(0.0 /*cvg*/);
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

    // TODO: pabandyti su fiksuotu g.
    fn set_exchange_interval(&mut self, cvg_avg: Float) {
        if cvg_avg >= 0.8 || cvg_avg <= 0.2 {
            // The formula gives the opposite result than is written in the article: when
            // convergence is low (very converged), it increases g.
            // Original formula.
            // let new_g = self.g as Float + ((0.5 - cvg_avg) * self.k as Float);
            // My formula.
            let new_g = self.g as Float + ((cvg_avg - 0.5) * self.k as Float);
            self.g = max(new_g as u32, 1);

            if self.mpi.is_root && self.g > 1 {
                dbg!(self.g);
            }
        }
        // eprintln!("rank: {}, next exchange in {} iters", self.mpi.rank, self.g);
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
                let distance = best_tour_with_hacks_appended_1
                    .without_hacks()
                    .distance(best_tour_with_hacks_appended_2.without_hacks());
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
        // if self.mpi.is_root {
        // eprintln!("proc_dist: {:?}", proc_distances);
        // }
        for y in 0..self.mpi.world_size {
            let row = proc_distances.row_mut(y);
            row.sort_unstable();

            // First element is going to be 0 since it is distance to self.
            let neighbour = Float::from(row[1..].iter().take(self.lowercase_q).sum::<u16>())
                / self.lowercase_q as Float;
            neighbour_values.push(neighbour);
        }
        // if self.mpi.is_root {
        // eprintln!("neighbour values: {neighbour_values:?}");
        // }
        neighbour_values
    }

    // Returns rank of the chosen CPU.
    fn select_exchange_partner(
        &mut self,
        fitness: &[Float],
        neighbour_values: &[Float],
        other_cpus: &[usize],
    ) -> usize {
        // Denominator might be zero, but it shouldn't happen with != 2 cpus.
        if self.mpi.world_size == 2 {
            return other_cpus[0];
        }
        let mut denominator = 0.0;
        let self_neighbour = neighbour_values[self.mpi.rank];
        for i in 0..neighbour_values.len() {
            if i == self.mpi.rank {
                continue;
            }
            // dbg!(self_neighbour, neighbour_values[i], fitness[i]);
            denominator += Float::abs(self_neighbour - neighbour_values[i]) * fitness[i];
        }
        // In the unlikely event that denominator == 0.0, return random CPU.
        if denominator == 0.0 {
            return *other_cpus.choose(self.rng).unwrap();
        }

        *other_cpus
            .choose_weighted(self.rng, |&cpu| {
                let numerator = (Float::abs(self_neighbour - neighbour_values[cpu]) * fitness[cpu]);
                // dbg!(numerator, denominator);
                numerator / denominator
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

    pub fn choose_except<'b>(
        tours: &'b [TourExt],
        distrib: Uniform<usize>,
        except: usize,
        rng: &mut R,
    ) -> (&'b Tour, &'b Tour) {
        let except_t = &tours[except];
        // loop {
        // let number = distrib.sample(rng);
        // We must not choose identical tours. Checking if they are identical is expensive,
        // so disallow all tours of the same length.
        // if number != except {
        // let t = &tours[number];
        // if t.length() != except_t.length() {
        let t = &tours[0]; // whatever, it is not used anyway
        return (except_t, t);
        // }
        // }
        // }
    }
}
