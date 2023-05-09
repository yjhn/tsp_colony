//! Combinatorial artificial bee colony.

use rand::Rng;

use crate::{
    bee::Bee,
    config::{DistanceT, Float},
    index::CityIndex,
    matrix::Matrix,
    tour::Tour,
    tsp_problem::TspProblem,
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
        let fitness = Self::fitness(&tour);
        Self {
            tour,
            non_improvement_iters: 0,
            fitness,
            prob_select_by_onlooker: 0.0,
        }
    }

    fn fitness(tour: &Tour) -> Float {
        1.0 / (1.0 + tour.length() as Float)
    }

    pub fn calc_set_prob_select(&mut self, fit_best: Float) {
        self.prob_select_by_onlooker = (0.9 * self.fitness) / fit_best + 0.1;
    }

    pub fn tour(&self) -> &Tour {
        &self.tour
    }

    pub fn set_tour(&mut self, tour: Tour) {
        self.fitness = Self::fitness(&tour);
        self.tour = tour;
        self.non_improvement_iters = 0;
    }

    pub fn inc_non_improvement(&mut self) {
        self.non_improvement_iters += 1;
    }

    pub fn non_improvement_iters(&self) -> u32 {
        self.non_improvement_iters
    }
}

pub struct CombArtBeeColony<'a, R: Rng> {
    bees: Vec<Bee>,
    iteration: u32, // max iterations will be specified on method evolve_until_optimal
    tours: Vec<TourExt>,
    best_tour: Tour,
    tour_non_improvement_limit: u32,
    tsp_problem: &'a TspProblem,
    // Row i is neighbour list for city CityIndex(i).
    neighbour_lists: NeighbourMatrix,
    rng: &'a mut R,
}

impl<'a, R: Rng> CombArtBeeColony<'a, R> {
    pub fn new(
        tsp_problem: &'a TspProblem,
        colony_size: u32,
        nl_max: u16,
        capital_l: Float,
        rng: &'a mut R,
    ) -> Self {
        let mut tours = Vec::with_capacity(colony_size as usize);
        let number_of_cities = tsp_problem.number_of_cities() as u16;
        let mut best_tour_idx = 0;
        let mut best_tour_length = DistanceT::MAX;
        for i in 0..colony_size {
            let tour = Tour::random(number_of_cities, tsp_problem.distances(), rng);
            if tour.length() < best_tour_length {
                best_tour_length = tour.length();
                best_tour_idx = i;
            }
            tours.push(TourExt::new(tour));
        }
        let best_tour = tours[best_tour_idx as usize].tour().clone();
        let neighbour_lists = tsp_problem.distances().neighbourhood_lists(nl_max);
        // Eq. 6 in qCABC paper.
        let tour_non_improvement_limit =
            (colony_size as Float * number_of_cities as Float) / capital_l;

        Self {
            bees: (0..colony_size).map(|_| Bee::new()).collect(),
            iteration: 0,
            tours,
            best_tour,
            tour_non_improvement_limit: tour_non_improvement_limit as u32,
            tsp_problem,
            neighbour_lists,
            rng,
        }
    }

    pub fn colony_size(&self) -> usize {
        self.bees.len()
    }

    pub fn iterate_until_optimal(&mut self, max_iterations: u32) -> bool {
        let num_cities = self.number_of_cities();
        // TODO: what is the split between employed bees and foragers (onlookers)?
        // For now we will assume that all bees are both foragers and onlookers.
        let mut found_optimal = false;

        loop {
            // Employed bees phase.
            for bee in self.bees.iter() {
                // bee.
            }

            // Onlooker (forager) bees phase.

            // Scout phase.

            self.iteration += 1;

            if self.iteration == max_iterations {}

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
}
