use rand::{seq::SliceRandom, Rng, SeedableRng};

use crate::{
    distance_matrix::DistanceMatrix,
    matrix::SquareMatrix,
    pheromone_visibility_matrix::{self, PheromoneVisibilityMatrix},
    tour::CityIndex,
    utils::{all_cities, order},
};

pub struct Ant {
    unvisited_cities: Vec<CityIndex>,
    tour: Vec<CityIndex>,
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
            current_city: starting_city,
        }
    }

    pub fn reset_to_city(&mut self, city_count: usize, starting_city: CityIndex) {
        self.unvisited_cities = all_cities(city_count);
        self.current_city = starting_city;
    }

    pub fn update_pheromones(&self, pheromone_matrix: &PheromoneVisibilityMatrix) {
        todo!()
    }

    // Sums pheromone levels raised to alpha * distances raised to beta
    // for all unvisited cities (divisor in formula 4 in the paper).
    // beta MUST BE negative
    pub fn sum_tau(&self, pheromone_matrix: &PheromoneVisibilityMatrix, alpha: f32) -> f32 {
        self.unvisited_cities
            .iter()
            .copied()
            .map(|city| {
                let ord = order(city, self.current_city);
                pheromone_matrix.get_pheromone(ord).powf(alpha)
                    * pheromone_matrix.get_visibility(ord)
            })
            .sum()
    }

    // This function does not check if the city has already been visited.
    pub fn choose_next_city<R: Rng + SeedableRng>(
        &mut self,
        rng: &mut R,
        pheromone_matrix: &PheromoneVisibilityMatrix,
        alpha: f32,
    ) -> f32 {
        self.unvisited_cities.shuffle(rng);
        // TODO: 4 formulė straipsnyje
        let divisor = self.sum_tau(pheromone_matrix, alpha);
        // TODO: gal pasidaryti ir čia uniform distribution? Bet neaišku, kiek kartų bus naudojama...

        for city in self.unvisited_cities.iter().copied() {
            let ord = order(city, self.current_city);
            let dividend = pheromone_matrix.get_pheromone(ord).powf(alpha)
                * pheromone_matrix.get_visibility(ord);
            let p = dividend / divisor;
        }

        todo!()
    }
}
