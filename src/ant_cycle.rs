use std::time::Instant;

use rand::{distributions, prelude::Distribution, Rng, SeedableRng};

use crate::{
    ant::Ant,
    config::Float,
    matrix::SquareMatrix,
    pheromone_visibility_matrix::PheromoneVisibilityMatrix,
    tour::{CityIndex, Tour},
    tsp_problem::TspProblem,
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
    q: Float,
    initial_trail_intensity: Float,
}

impl<'a, R: Rng + SeedableRng> AntCycle<'a, R> {
    pub fn new(
        ant_count: usize,
        mut rng: &'a mut R,
        tsp_problem: &'a TspProblem,
        initial_trail_intensity: Float,
        alpha: Float,
        beta: Float,
        q: Float,
        ro: Float,
    ) -> AntCycle<'a, R> {
        let city_count = tsp_problem.number_of_cities();
        let pheromone_matrix = PheromoneVisibilityMatrix::new(
            city_count,
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
            q,
            initial_trail_intensity,
        }
    }

    pub fn set_alpha(&mut self, alpha: Float) {
        self.alpha = alpha;
    }

    pub fn set_q(&mut self, q: Float) {
        self.q = q;
    }

    pub fn set_ro(&mut self, ro: Float) {
        self.pheromone_matrix.set_ro(ro);
    }

    /// Resets pheromone intensity to `intensity`.
    pub fn reset_pheromone(&mut self, intensity: Float) {
        self.pheromone_matrix.reset_pheromone(intensity);
    }

    fn reset_all_state(&mut self) {
        // Reset ants.
        for a in self.ants.iter_mut() {
            let num_cities = self.tsp_problem.number_of_cities();
            let mut distrib = distributions::Uniform::new(0, num_cities).unwrap();
            a.reset_to_city(num_cities, CityIndex::new(distrib.sample(&mut self.rng)));
        }
        self.time = 0;
        self.iteration = 0;
        self.best_tour = Tour::PLACEHOLDER;
        self.pheromone_matrix
            .reset_pheromone(self.initial_trail_intensity);
    }

    pub fn iterate_until_optimal(&mut self, max_iterations: u32) -> bool {
        let num_cities = self.tsp_problem.number_of_cities();
        let mut distrib = distributions::Uniform::new(0, num_cities).unwrap();

        struct ShortestIterationTour {
            ant_idx: usize,
            tour_length: u32,
        }
        let mut found_optimal = false;

        loop {
            dbg!(self.iteration);
            dbg!(self.best_tour.length());
            let t = Instant::now();
            // TODO: gal precomputint pheromone.powf(alpha)? Vis tiek jis keičiasi tik tik kitoje iteracijoje.
            // Each ant constructs a tour, keep track of the shortest tour found in this iteration.
            let mut short = ShortestIterationTour {
                ant_idx: 0,
                tour_length: u32::MAX,
            };

            // Colony is stale if all ants find the same tour. Since actually
            // checking if tours match is expensive, we only check if they are
            // of the same length.
            let mut stale = true;
            for (idx, ant) in self.ants.iter_mut().enumerate() {
                for _ in 1..num_cities {
                    ant.choose_next_city(self.rng, &self.pheromone_matrix, self.alpha);
                }

                let len = ant.tour_length(self.tsp_problem.distances());
                // dbg!(len);
                if len < short.tour_length {
                    stale = false;
                    short = ShortestIterationTour {
                        ant_idx: idx,
                        tour_length: len,
                    };
                } else if len > short.tour_length {
                    stale = false;
                }
            }

            // Update pheromone.
            self.pheromone_matrix.evaporate_pheromone();
            for ant in self.ants.iter() {
                ant.update_pheromone(&mut self.pheromone_matrix, self.q);
            }

            // Keep track of the shortest tour.
            if short.tour_length < self.best_tour.length() {
                self.best_tour = self.ants[short.ant_idx].clone_tour();
            }

            if self.iteration == max_iterations {
                println!("Stopping, reached max iterations count");
                break;
            }
            if self.best_tour.length() == self.tsp_problem.solution_length() {
                println!(
                    "Stopping, reached optimal length in iteration {}",
                    self.iteration
                );
                found_optimal = true;
                break;
            }
            if stale {
                println!(
                    "Stopping, colony is stale (all ants likely found the same tour). Iteration {}",
                    self.iteration
                );
                break;
            }

            self.iteration += 1;
            self.time += num_cities;

            // Reset ants.
            for a in self.ants.iter_mut() {
                a.reset_to_city(num_cities, CityIndex::new(distrib.sample(&mut self.rng)));
            }

            dbg!(t.elapsed().as_nanos());
        }
        // Reset ants.
        for a in self.ants.iter_mut() {
            a.reset_to_city(num_cities, CityIndex::new(distrib.sample(&mut self.rng)));
        }
        self.reset_all_state();

        found_optimal
    }

    pub fn iteration(&self) -> u32 {
        self.iteration
    }

    pub fn best_tour(&self) -> &Tour {
        &self.best_tour
    }
}
