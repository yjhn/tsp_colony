use rand::{distributions::Uniform, prelude::Distribution, seq::SliceRandom, Rng, SeedableRng};

use crate::{
    config::Float,
    distance_matrix::DistanceMatrix,
    matrix::SquareMatrix,
    pheromone_visibility_matrix::{self, PheromoneVisibilityMatrix},
    tour::{CityIndex, Length, Tour},
    utils::{all_cities, all_cities_fill, order, reverse_order},
};

pub struct Ant {
    // TODO: maybe use one vec for both visited and unvisited cities.
    // Unvisited cities would be at the front, and the tour would grow from the back
    // or vice-versa.
    unvisited_cities: Vec<CityIndex>,
    tour: Vec<CityIndex>,
    // It would be cleaner to use Option<u32> here, but then it would have
    // to be unwrapped frequently.
    tour_length: u32,
    // TODO: maybe rmove current_city, since it is always the same as the last element in tour?
    current_city: CityIndex,
}

impl Ant {
    pub fn new(city_count: usize, starting_city: CityIndex) -> Ant {
        let mut unvisited_cities: Vec<CityIndex> = all_cities(city_count);
        unvisited_cities.swap_remove(starting_city.get());
        let tour = vec![starting_city];

        Ant {
            unvisited_cities,
            tour,
            tour_length: u32::MAX,
            current_city: starting_city,
        }
    }

    pub fn reset_to_city(&mut self, city_count: usize, starting_city: CityIndex) {
        self.tour.clear();
        self.tour_length = u32::MAX;
        all_cities_fill(&mut self.unvisited_cities, city_count);
        self.visit_city(starting_city, starting_city.get());
    }

    /// `idx` - index in unvisited cities `Vec`.
    fn visit_city(&mut self, city: CityIndex, idx: usize) {
        self.current_city = city;
        self.unvisited_cities.swap_remove(idx);
        self.tour.push(city);
    }

    pub fn update_pheromone(&self, pheromone_matrix: &mut PheromoneVisibilityMatrix, q: Float) {
        debug_assert!(self.tour_length < u32::MAX);

        let delta_tau = q / self.tour_length as Float;
        for path in self.tour.windows(2) {
            if let &[c1, c2] = path {
                pheromone_matrix.adjust_pheromone(reverse_order(c1, c2), delta_tau);
            }
        }
        // Last path.
        pheromone_matrix.adjust_pheromone(
            reverse_order(self.tour[0], *self.tour.last().unwrap()),
            delta_tau,
        );
    }

    /// Calculates and caches tour length.
    pub fn tour_length(&mut self, distances: &DistanceMatrix) -> u32 {
        if self.tour_length == u32::MAX {
            self.tour_length = self.tour.calculate_tour_length(distances)
        }
        self.tour_length
    }

    // Sums pheromone levels raised to alpha * distances raised to beta
    // for all unvisited cities (divisor in formula 4 in the paper).
    pub fn sum_tau(&self, matrix: &PheromoneVisibilityMatrix, alpha: Float) -> Float {
        self.unvisited_cities
            .iter()
            .copied()
            .map(|city| {
                let ord = order(city, self.current_city);
                matrix.pheromone(ord).powf(alpha) * matrix.visibility(ord)
            })
            .sum()
    }

    pub fn choose_next_city<R: Rng + SeedableRng>(
        &mut self,
        rng: &mut R,
        matrix: &PheromoneVisibilityMatrix,
        alpha: Float,
    ) {
        self.unvisited_cities.shuffle(rng);
        // TODO: 4 formulė straipsnyje
        let divisor = self.sum_tau(matrix, alpha);
        // TODO: gal pasidaryti ir čia uniform distribution? Bet neaišku, kiek kartų bus naudojama...
        let distrib = Uniform::new(0.0, 1.0).unwrap();

        for idx in 0..self.unvisited_cities.len() {
            let city = self.unvisited_cities[idx];
            let ord = order(city, self.current_city);
            let dividend = matrix.pheromone(ord).powf(alpha) * matrix.visibility(ord);
            let p = dividend / divisor;
            if p > distrib.sample(rng) {
                // This city is the chosen one.
                self.visit_city(city, idx);
                return;
            }
        }
    }

    /// Clones the ant's tour.
    pub fn clone_tour(&self) -> crate::tour::Tour {
        // Length must be already calculated.
        debug_assert!(self.tour_length != u32::MAX);

        Tour::clone_from_cities(&self.tour, self.tour_length)
    }
}
