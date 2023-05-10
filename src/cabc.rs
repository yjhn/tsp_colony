//! Combinatorial artificial bee colony.

use std::ops::{Deref, DerefMut};

use rand::{
    distributions::{Uniform, WeightedIndex},
    prelude::Distribution,
    seq::SliceRandom,
    Rng,
};

use crate::{
    config::{DistanceT, Float},
    gstm,
    index::CityIndex,
    matrix::{Matrix, SquareMatrix},
    path_usage_matrix::PathUsageMatrix,
    tour::{Tour, TourFunctions},
    tsp_problem::TspProblem,
    utils::choose_except,
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
        1.0 / (1.0 + tour.length() as Float)
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

// impl DerefMut for TourExt {
//     fn deref_mut(&mut self) -> &mut Self::Target {
//         &mut self.tour
//     }
// }

pub struct CombArtBeeColony<'a, R: Rng> {
    colony_size: usize,
    iteration: u32, // max iterations will be specified on method evolve_until_optimal
    tours: Vec<TourExt>,
    best_tour: Tour,
    tour_non_improvement_limit: u32,
    tsp_problem: &'a TspProblem,
    // Row i is neighbour list for city CityIndex(i).
    neighbour_lists: NeighbourMatrix,
    path_usage_matrix: PathUsageMatrix, // used for parallelization only
    p_cp: Float,
    p_rc: Float,
    p_l: Float,
    r: Float,
    rng: &'a mut R,
}

impl<'a, R: Rng> CombArtBeeColony<'a, R> {
    pub fn new(
        tsp_problem: &'a TspProblem,
        colony_size: usize,
        nl_max: u16,
        capital_l: Float,
        p_cp: Float,
        p_rc: Float,
        p_l: Float,
        r: Float,
        rng: &'a mut R,
    ) -> Self {
        let mut tours = Vec::with_capacity(colony_size);
        let number_of_cities = tsp_problem.number_of_cities() as u16;
        let mut path_usage_matrix = PathUsageMatrix::new(number_of_cities);

        let mut best_tour_idx = 0;
        let mut best_tour_length = DistanceT::MAX;
        for i in 0..colony_size {
            let tour = Tour::random(number_of_cities, tsp_problem.distances(), rng);
            if tour.length() < best_tour_length {
                best_tour_length = tour.length();
                best_tour_idx = i;
            }
            path_usage_matrix.inc_tour_paths(&tour);
            tours.push(TourExt::new(tour));
        }
        let neighbour_lists = tsp_problem.distances().neighbourhood_lists(nl_max);
        // Eq. 6 in qCABC paper.
        let tour_non_improvement_limit =
            (colony_size as Float * Float::from(number_of_cities)) / capital_l;
        let best_tour = tours[best_tour_idx].tour().clone();

        Self {
            colony_size,
            iteration: 0,
            tours,
            best_tour,
            tour_non_improvement_limit: tour_non_improvement_limit as u32,
            tsp_problem,
            neighbour_lists,
            path_usage_matrix,
            p_cp,
            p_rc,
            p_l,
            r,
            rng,
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

        loop {
            // Employed bees phase.
            self.employed_bees_phase(distrib_tours);

            // Onlooker (forager) bees phase.
            self.onlooker_bees_phase(&mut tour_distances, distrib_tours);

            // Scout phase.
            self.scout_bees_phase();

            self.iteration += 1;

            if self.best_tour.length() == self.tsp_problem.solution_length() {
                eprintln!(
                    "Stopping, reached optimal length in iteration {}",
                    self.iteration
                );
                found_optimal = true;
                break;
            }

            if self.iteration == max_iterations {
                eprintln!("Stopping, reached max iterations count");
                break;
            }
        }

        found_optimal
    }

    pub fn number_of_cities(&self) -> usize {
        self.tsp_problem.number_of_cities()
    }

    /// Mean distance of `x_i` to all the other tours.
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
                self.rng,
                self.p_rc,
                self.p_cp,
                self.p_l,
                &self.neighbour_lists,
                self.tsp_problem.distances(),
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
                self.rng,
                self.p_rc,
                self.p_cp,
                self.p_l,
                &self.neighbour_lists,
                self.tsp_problem.distances(),
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
            }
        }
        // New best could be longer, because if the best tour is not improved
        // for some generations, it is replaced.
        if best_tour_length < self.best_tour.length() {
            self.best_tour = self.tours[best_tour_idx].tour().clone();
        }
    }
}
