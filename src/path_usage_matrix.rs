use std::ops::Index;

use crate::{
    config::Float, index::CityIndex, matrix::SquareMatrix, tour::TourFunctions,
    utils::reverse_order,
};

pub struct PathUsageMatrix(SquareMatrix<u16>);

impl PathUsageMatrix {
    pub fn new(number_of_cities: u16) -> Self {
        Self(SquareMatrix::new(usize::from(number_of_cities), 0))
    }

    pub fn inc_tour_paths(&mut self, tour: &[CityIndex]) {
        for path in tour.paths() {
            let &[c1, c2] = path else {unreachable!()};
            self.0[reverse_order(usize::from(c1), usize::from(c2))] += 1;
            // self.0[(usize::from(c2), usize::from(c1))] += 1;
        }
        // Last path.
        let last_c = usize::from(*tour.last().unwrap());
        self.0[reverse_order(usize::from(tour[0]), last_c)] += 1;
        // self.0[(last_c, usize::from(tour[0]))] += 1;
    }

    pub fn dec_tour_paths(&mut self, tour: &[CityIndex]) {
        for path in tour.paths() {
            let &[c1, c2] = path else {unreachable!()};
            // debug_assert_ne!(self.0[(usize::from(c1), usize::from(c2))], 0);
            self.0[reverse_order(usize::from(c1), usize::from(c2))] -= 1;
            // self.0[(usize::from(c2), usize::from(c1))] -= 1;
        }
        // Last path.
        let last_c = usize::from(*tour.last().unwrap());
        self.0[reverse_order(usize::from(tour[0]), last_c)] -= 1;
        // self.0[(last_c, usize::from(tour[0]))] -= 1;
    }

    pub fn max(&self) -> u16 {
        let mut max = 0;
        for x in 0..self.0.side_length() {
            for y in 0..x {
                let elem = self.0[(x, y)];
                if elem > max {
                    max = elem;
                }
            }
        }

        max
    }

    pub fn max_avg(&self) -> (u16, Float) {
        let mut max = 0;
        let mut sum: u32 = 0;
        for x in 0..self.0.side_length() {
            for y in 0..x {
                let elem = self.0[(x, y)];
                sum += u32::from(elem);
                if elem > max {
                    max = elem;
                }
            }
        }

        // TODO: f32 might really lose precision here.
        let avg = sum as Float / self.unique_path_count() as Float;

        (max, avg)
    }

    pub fn convergence(&self) -> Float {
        let (max, avg) = self.max_avg();
        let first_part = 1.0 / (self.unique_path_count() as Float * (Float::from(max) - avg));

        let mut sum = 0.0;
        for x in 0..self.0.side_length() {
            for y in 0..x {
                let elem = self.0[(x, y)];
                sum += Float::abs(Float::from(elem) - avg);
            }
        }

        first_part * sum
    }

    pub fn unique_path_count(&self) -> usize {
        (self.0.side_length() * (self.0.side_length() - 1)) / 2
    }
}

// impl Index<(CityIndex, CityIndex)> for PathUsageMatrix {
// type Output = u16;

// fn index(&self, (x, y): (CityIndex, CityIndex)) -> &Self::Output {
// self.0.index((usize::from(x), usize::from(y)))
// }
// }
