use rand::{seq::SliceRandom, Rng, SeedableRng};

use crate::{
    pheromone_matrix::{self, PheromoneMatrix},
    tour::CityIndex,
    utils::all_cities,
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
    }

    pub fn choose_next_city<R: Rng + SeedableRng>(
        &mut self,
        rng: &mut R,
        pheromone_matrix: &PheromoneMatrix,
    ) {
        self.unvisited_cities.shuffle(rng);
        todo!();
        // Choose a city, remove it from unvisited, add to tour.
        self.current_city = todo!();
    }

    pub fn update_pheromones(&self, pheromone_matrix: &PheromoneMatrix) {
        todo!()
    }

    // This function does not check if the city has already been visited.
    pub fn transition_probability(&self, j: CityIndex, pheromone_matrix: &PheromoneMatrix) -> f32 {
        // TODO: 4 formulė straipsnyje
    }
}
