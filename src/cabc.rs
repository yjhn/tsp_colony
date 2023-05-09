//! Combinatorial artificial bee colony.

use rand::Rng;

use crate::{
    bee::Bee, config::DistanceT, index::CityIndex, matrix::Matrix, tour::Tour,
    tsp_problem::TspProblem,
};

pub type NeighbourMatrix = Matrix<(CityIndex, DistanceT)>;

pub struct CombArtBeeColony<'a, R: Rng> {
    bees: Vec<Bee>,
    iteration: u32, // max iterations will be specified on method evolve_until_optimal
    tours: Vec<(Tour, u32)>, // u32 is for counting non-improvment iterations
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
        tour_non_improvement_limit: u32,
        nl_max: u16,
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
            tours.push((tour, 0));
        }
        let best_tour = tours[best_tour_idx as usize].0.clone();
        let neighbour_lists = tsp_problem.distances().neighbourhood_lists(nl_max);

        Self {
            bees: (0..colony_size).map(|_| Bee::new()).collect(),
            iteration: 0,
            tours,
            best_tour,
            tour_non_improvement_limit,
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
            // for bee in

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

        return found_optimal;
    }

    pub fn number_of_cities(&self) -> usize {
        self.tsp_problem.number_of_cities()
    }
}
