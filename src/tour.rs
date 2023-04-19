use crate::matrix::SquareMatrix;
use crate::utils::all_cities;
use rand::prelude::SliceRandom;
use rand::Rng;
use std::fmt::Display;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::slice::Windows;

/// Index of the city in the city matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CityIndex(usize);

impl CityIndex {
    pub fn new(index: usize) -> CityIndex {
        CityIndex(index)
    }

    pub fn get(self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone)]
pub struct Tour {
    cities: Vec<CityIndex>,
    tour_length: u32,
}

impl Tour {
    pub const PLACEHOLDER: Tour = Tour {
        cities: Vec::new(),
        tour_length: u32::MAX,
    };

    pub fn number_of_cities(&self) -> usize {
        self.cities.len()
    }

    pub fn length(&self) -> u32 {
        self.tour_length
    }

    pub fn cities(&self) -> &[CityIndex] {
        &self.cities
    }

    pub fn from_cities(cities: Vec<CityIndex>, distances: &SquareMatrix<u32>) -> Tour {
        let tour_length = cities.calculate_tour_length(distances);

        Tour {
            cities,
            tour_length,
        }
    }

    pub fn from_hack_cities(mut cities_with_length: Vec<CityIndex>) -> Tour {
        let tour_length_usize = cities_with_length.pop().unwrap().get();
        let tour_length = tour_length_usize as u32;
        // Cities don't contain length anymore.
        let cities = cities_with_length;

        Tour {
            cities,
            tour_length,
        }
    }

    pub fn random(city_count: usize, distances: &SquareMatrix<u32>, rng: &mut impl Rng) -> Tour {
        assert!(city_count > 1);

        let mut cities = all_cities(city_count);
        cities.shuffle(rng);
        let tour_length = cities.calculate_tour_length(distances);

        Tour {
            cities,
            tour_length,
        }
    }

    // This function call must be matched by the corresponding
    // call to remove_hack_length().
    pub fn hack_append_length_at_tour_end(&mut self) {
        // This is only valid on >=32 bit architectures.
        let length_usize = self.tour_length as usize;
        let length_index = CityIndex::new(length_usize);
        self.cities.push(length_index);
    }

    pub fn remove_hack_length(&mut self) {
        self.cities.pop();
    }
    pub fn is_shorter_than(&self, other: &Tour) -> bool {
        self.tour_length < other.tour_length
    }

    pub fn save_to_file<P: AsRef<Path>, Dp: AsRef<Path> + Display>(
        &self,
        problem_path: Dp,
        path: P,
    ) {
        let file = File::create(path).unwrap();
        let mut file = BufWriter::new(file);

        writeln!(file, "Problem file: {problem_path}").unwrap();
        writeln!(file, "Number of cities: {}", self.number_of_cities()).unwrap();
        writeln!(file, "Tour length: {}", self.tour_length).unwrap();
        writeln!(file, "Cities:").unwrap();

        // Use indices starting at 1 for output, same as TSPLIB
        // format for consistency.
        for city in &self.cities {
            write!(file, "{} ", city.get() + 1).unwrap();
        }
        writeln!(file).unwrap();
    }

    pub fn nearest_neighbour(
        city_count: usize,
        starting_city: Option<usize>,
        distances: &SquareMatrix<u32>,
        rng: &mut impl Rng,
    ) -> Tour {
        let starting_city = if let Some(c) = starting_city {
            c
        } else {
            rng.gen_range(0..city_count)
        };

        let mut cities = Vec::with_capacity(city_count);

        // Still unused cities.
        let mut unused_cities: Vec<usize> = (0..city_count).collect();
        unused_cities.swap_remove(starting_city);

        let mut tour_length = 0;
        let mut last_added_city = starting_city;

        while !unused_cities.is_empty() {
            let mut min_distance = u32::MAX;
            let mut min_distance_city = unused_cities[0];
            let mut min_distance_city_idx = 0;
            for i in 0..unused_cities.len() {
                let c = unused_cities[i];
                let dist = distances[(last_added_city, c)];
                if dist < min_distance {
                    min_distance = dist;
                    min_distance_city = c;
                    min_distance_city_idx = i;
                }
            }

            // Insert.
            cities.push(CityIndex::new(min_distance_city));
            last_added_city = min_distance_city;
            tour_length += min_distance;
            unused_cities.swap_remove(min_distance_city_idx);
        }

        tour_length += distances[(last_added_city, starting_city)];

        Tour {
            cities,
            tour_length,
        }
    }

    pub fn last_to_first_path(&self) -> (CityIndex, CityIndex) {
        (*self.cities.last().unwrap(), self.cities[0])
    }

    // Returns iterator over all paths except for the
    // first -> last.
    pub fn paths(&self) -> Windows<'_, CityIndex> {
        self.cities.windows(2)
    }
}

pub trait Length {
    fn calculate_tour_length(&self, distances: &SquareMatrix<u32>) -> u32;

    fn hack_get_tour_length_from_last_element(&self) -> u32;
}

impl Length for [CityIndex] {
    fn calculate_tour_length(&self, distances: &SquareMatrix<u32>) -> u32 {
        assert!(self.len() > 1);

        let mut tour_length = 0;
        for idx in 1..self.len() {
            tour_length += distances[(self[idx - 1].get(), self[idx].get())];
        }
        // Add distance from last to first.
        tour_length += distances[(self.last().unwrap().get(), self[0].get())];

        tour_length
    }

    // The length must first be inserted using Tour::hack_append_length_at_tour_end().
    fn hack_get_tour_length_from_last_element(&self) -> u32 {
        let tour_length_usize = self.last().unwrap().get();

        tour_length_usize as u32
    }
}
