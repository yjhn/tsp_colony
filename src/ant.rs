use rand::{distributions::Uniform, prelude::Distribution, seq::SliceRandom, Rng, SeedableRng};

use crate::{
    config::Float,
    distance_matrix::DistanceMatrix,
    matrix::SquareMatrix,
    pheromone_visibility_matrix::PheromoneVisibilityMatrix,
    tour::{CityIndex, Tour, TourFunctions},
    utils::{all_cities, all_cities_fill, order, reverse_order},
};
use iterator_ilp::IteratorILP;

pub struct Ant {
    // TODO: maybe use one vec for both visited and unvisited cities.
    // Unvisited cities would be at the front, and the tour would grow from the back
    // or vice-versa.
    // Both tour and unvisited_cities have fixed capacity once the ant is created.
    unvisited_cities: Vec<CityIndex>,
    tour: Vec<CityIndex>,
    // It would be cleaner to use Option<u32> here, but then it would have
    // to be unwrapped frequently.
    tour_length: u32,
    // TODO: maybe remove current_city, since it is always the same as the last element in tour?
    current_city: CityIndex,
}

impl Ant {
    pub fn new(city_count: u16, starting_city: CityIndex) -> Ant {
        let mut unvisited_cities: Vec<CityIndex> = all_cities(city_count);
        unvisited_cities.swap_remove(starting_city.into());
        let tour = vec![starting_city];

        Ant {
            unvisited_cities,
            tour,
            tour_length: u32::MAX,
            current_city: starting_city,
        }
    }

    pub fn reset_to_city(&mut self, city_count: u16, starting_city: CityIndex) {
        self.tour.clear();
        self.tour_length = u32::MAX;
        all_cities_fill(&mut self.unvisited_cities, city_count);
        self.visit_city(starting_city, starting_city.into());
    }

    /// `idx` - index in unvisited cities `Vec`.
    fn visit_city(&mut self, city: CityIndex, idx: usize) {
        self.current_city = city;
        self.unvisited_cities.swap_remove(idx);
        self.tour.push(city);
    }

    pub fn update_pheromone(&self, delta_tau_matrix: &mut SquareMatrix<Float>, capital_q: Float) {
        debug_assert!(self.tour_length < u32::MAX);

        // TODO: should Q/L_k be Q/L_l? (current implementation assumes it)
        let delta_tau = capital_q / self.tour_length as Float;
        self.tour.update_pheromone(delta_tau_matrix, delta_tau);
    }

    /// Calculates and caches tour length.
    pub fn tour_length(&mut self, distances: &DistanceMatrix) -> u32 {
        // dbg!(self.tour.len());
        if self.tour_length == u32::MAX {
            self.tour_length = self.tour.calculate_tour_length(distances);
        }
        self.tour_length
    }

    // Sums pheromone levels raised to alpha * visibilities
    // for all unvisited cities (divisor in formula 4 in the paper).
    pub fn sum_tau(&self, matrix: &PheromoneVisibilityMatrix, alpha: Float) -> Float {
        self.unvisited_cities
            .iter()
            .copied()
            .map(|city| {
                let ord = order(city, self.current_city);
                // matrix.pheromone((ord.1, ord.0)).powf(alpha) * matrix.visibility(ord)
                matrix.pheromone((ord.1, ord.0)) * matrix.visibility(ord)
            })
            .sum()
        // .sum_ilp::<8, _>()
    }

    pub fn choose_next_city<R: Rng + SeedableRng>(
        &mut self,
        rng: &mut R,
        distrib: &Uniform<Float>,
        matrix: &PheromoneVisibilityMatrix,
        alpha: Float,
    ) {
        debug_assert!(!self.unvisited_cities.is_empty());

        self.unvisited_cities.shuffle(rng);
        // TODO: 4 formulė straipsnyje
        let denominator = self.sum_tau(matrix, alpha);
        // debug_assert!(denominator > 0.0, "denominator: {denominator}");
        if denominator <= 0.0 {
            // If denominator is 0, all paths pheromone is 0,
            // so choose any city.
            self.visit_city(self.unvisited_cities[0], 0);
            return;
        }
        // let denominator = if denominator > 0.0 {
        //     denominator
        // } else {
        //     Float::MIN_POSITIVE
        // };

        let mut max_prob_city_idx = 0;
        let mut max_prob = -1.0;
        let mut max_prob_city = CityIndex::new(0);
        for idx in 0..self.unvisited_cities.len() {
            let city = self.unvisited_cities[idx];
            let ord = order(city, self.current_city);
            // let numerator = matrix.pheromone((ord.1, ord.0)).powf(alpha) * matrix.visibility(ord);
            let numerator = matrix.pheromone((ord.1, ord.0)) * matrix.visibility(ord);
            let p = numerator / denominator;
            debug_assert!(p <= 1.0 && p >= 0.0, "p: {}", p);
            if p > distrib.sample(rng) {
                // This city is the chosen one.
                self.visit_city(city, idx);
                return;
            } else if p > max_prob {
                max_prob = p;
                max_prob_city_idx = idx;
                max_prob_city = city;
            }
        }
        debug_assert!(max_prob >= 0.0, "max_prob: {max_prob}");
        // We have to choose a city in this function, so choose the first
        // in unvisited if the loop fails to select any.
        // Taken from:
        // https://github.com/ppoffice/ant-colony-tsp/blob/d4e6dfb880728fe2de8ac59b723879b0e662ad0c/aco.py#L96-L107
        // TODO: find a better solution.
        // self.visit_city(self.unvisited_cities[0], 0);
        self.visit_city(max_prob_city, max_prob_city_idx);
    }

    /// Clones the ant's tour.
    pub fn clone_tour(&self, distances: &DistanceMatrix) -> Tour {
        debug_assert_eq!(self.unvisited_cities.len(), 0);
        // Length must be already calculated.
        debug_assert!(self.tour_length != u32::MAX);

        Tour::clone_from_cities(&self.tour, self.tour_length, distances)
    }
}
