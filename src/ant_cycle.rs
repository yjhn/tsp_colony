use std::{cmp::max, time::Instant};

use mpi::traits::CommunicatorCollectives;
use rand::{distributions, prelude::Distribution, Rng, SeedableRng};

use crate::{
    ant::Ant,
    config::Float,
    matrix::SquareMatrix,
    pheromone_visibility_matrix::PheromoneVisibilityMatrix,
    tour::{CityIndex, Tour},
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
    mpi: &'a Mpi<'a>,
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
        mpi: &'a Mpi,
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
            capital_q_mul,
            initial_trail_intensity,
            mpi,
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
        let num_cities = self.tsp_problem.number_of_cities();
        let mut distrib = distributions::Uniform::new(0, num_cities).unwrap();
        for a in self.ants.iter_mut() {
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
        let distrib01 = distributions::Uniform::new(0.0, 1.0).unwrap();

        struct ShortestIterationTour {
            ant_idx: usize,
            tour_length: u32,
        }
        let mut found_optimal = false;
        let mut delta_tau_matrix = SquareMatrix::new(num_cities, 0.0);

        loop {
            // let t = Instant::now();
            // TODO: gal precomputint pheromone.powf(alpha)? Vis tiek jis keičiasi tik kitoje iteracijoje.
            // Each ant constructs a tour, keep track of the shortest tour found in this iteration.
            let mut short = ShortestIterationTour {
                ant_idx: 0,
                tour_length: u32::MAX,
            };
            // Also keep track of the longest tour to calculate Q.
            let mut longest_tour_length = 0;

            // Colony is stale if all ants find the same tour. Since actually
            // checking if tours match is expensive, we only check if they are
            // of the same length.
            let mut stale = true;
            for (idx, ant) in self.ants.iter_mut().enumerate() {
                for _ in 1..num_cities {
                    ant.choose_next_city(self.rng, &distrib01, &self.pheromone_matrix, self.alpha);
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
                if len > longest_tour_length {
                    longest_tour_length = len;
                }
            }

            // Update pheromone.
            self.pheromone_matrix.evaporate_pheromone();

            let capital_q = self.capital_q_mul * longest_tour_length as Float;
            // TODO: figure out if this should be calculated using best length so far or only from this iteration.
            let min_tau = capital_q / short.tour_length as f32;

            for ant in self.ants.iter() {
                ant.update_pheromone(&mut delta_tau_matrix, self.capital_q_mul);
            }

            for x in 0..num_cities {
                for y in 0..x {
                    let delta = delta_tau_matrix[(x, y)];
                    self.pheromone_matrix.adjust_pheromone(
                        (CityIndex::new(x), CityIndex::new(y)),
                        if delta > min_tau { delta } else { min_tau },
                    );
                }
            }

            delta_tau_matrix.fill(0.0);
            // Keep track of the shortest tour.
            if short.tour_length < self.best_tour.length() {
                self.best_tour = self.ants[short.ant_idx].clone_tour(self.tsp_problem.distances());
                dbg!(self.iteration, self.best_tour.length());
            }

            self.iteration += 1;
            self.time += num_cities;

            // TODO: use all_gather_into for actual information exchange:
            // first, all processors exchange info on the routes of their best routes so far
            // then, each processor chooses with which processor to exchange info and does it
            let send_buf = [self.mpi.rank];
            let mut recv_buf = vec![0; self.mpi.world_size as usize];
            self.mpi.world.all_gather_into(&send_buf, &mut recv_buf);
            if self.mpi.is_root {
                dbg!(recv_buf);
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

            // Reset ants.
            for a in self.ants.iter_mut() {
                a.reset_to_city(num_cities, CityIndex::new(distrib.sample(&mut self.rng)));
            }

            // dbg!(t.elapsed().as_nanos());
        }

        found_optimal
    }

    pub fn iteration(&self) -> u32 {
        self.iteration
    }

    pub fn best_tour(&self) -> &Tour {
        &self.best_tour
    }
}
