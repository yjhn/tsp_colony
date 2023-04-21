use rand::{distributions, prelude::Distribution, Rng, SeedableRng};

use crate::{
    ant::Ant,
    matrix::SquareMatrix,
    pheromone_visibility_matrix::PheromoneVisibilityMatrix,
    tour::{CityIndex, Tour},
    tsp_problem::TspProblem,
};

pub struct AntCycle<R: Rng + SeedableRng> {
    time: u32,
    iteration: u32,
    ant_count: usize,
    ants: Vec<Ant>,
    rng: R,
    best_tour: Tour,
    pheromone_matrix: PheromoneVisibilityMatrix,
    tsp_problem: TspProblem,
    alpha: f32,
}

impl<R: Rng + SeedableRng> AntCycle<R> {
    pub fn new(
        ant_count: usize,
        mut rng: R,
        city_count: usize,
        tsp_problem: TspProblem,
        initial_trail_intensity: f32,
        alpha: f32,
        beta: f32,
    ) -> AntCycle<R> {
        let pheromone_matrix = PheromoneVisibilityMatrix::new(
            city_count,
            initial_trail_intensity,
            tsp_problem.distances(),
            beta,
        );
        let mut ants = Vec::with_capacity(ant_count);
        let distrib = distributions::Uniform::new(0, city_count)
            .unwrap()
            .sample_iter(&mut rng)
            .map(CityIndex::new);
        for (_, random_city) in (0..ant_count).zip(distrib) {
            let ant = Ant::new(city_count, random_city);
            ants.push(ant);
        }

        AntCycle {
            time: 0,
            iteration: 0,
            ant_count,
            ants: Vec::with_capacity(ant_count),
            rng,
            best_tour: Tour::PLACEHOLDER,
            pheromone_matrix,
            tsp_problem,
            alpha,
        }
    }

    pub fn iterate_until_optimal(&mut self, max_iterations: u32) {
        self.iteration += 1;
        for c in 0..self.tsp_problem.number_of_cities() {
            for a in self.ants.iter_mut() {
                a.choose_next_city(&mut self.rng, &self.pheromone_matrix, self.alpha);
            }
        }

        let num_cities = self.tsp_problem.number_of_cities();
        let mut distrib = distributions::Uniform::new(0, num_cities).unwrap();
        loop {
            self.iteration += 1;
            if self.iteration == max_iterations || todo!() {
                break;
            }
            for a in self.ants.iter_mut() {
                a.reset_to_city(num_cities, CityIndex::new(distrib.sample(&mut self.rng)));
            }

            for c in 1..self.tsp_problem.number_of_cities() {
                for a in self.ants.iter_mut() {
                    a.choose_next_city(&mut self.rng, &self.pheromone_matrix, self.alpha);
                }
            }
        }
    }
}
