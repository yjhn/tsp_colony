use std::ops::Index;

use crate::{index::CityIndex, matrix::SquareMatrix, tour::TourFunctions};

pub struct PathUsageMatrix(SquareMatrix<u16>);

impl PathUsageMatrix {
    pub fn new(number_of_cities: u16) -> Self {
        Self(SquareMatrix::new(usize::from(number_of_cities), 0))
    }

    pub fn inc_tour_paths(&mut self, tour: &[CityIndex]) {
        for path in tour.paths() {
            let &[c1, c2] = path else {unreachable!()};
            self.0[(usize::from(c1), usize::from(c2))] += 1;
            self.0[(usize::from(c2), usize::from(c1))] += 1;
        }
        // Last path.
        let last_c = usize::from(*tour.last().unwrap());
        self.0[(usize::from(tour[0]), last_c)] += 1;
        self.0[(last_c, usize::from(tour[0]))] += 1;
    }

    pub fn dec_tour_paths(&mut self, tour: &[CityIndex]) {
        for path in tour.paths() {
            let &[c1, c2] = path else {unreachable!()};
            self.0[(usize::from(c1), usize::from(c2))] -= 1;
            self.0[(usize::from(c2), usize::from(c1))] -= 1;
        }
        // Last path.
        let last_c = usize::from(*tour.last().unwrap());
        self.0[(usize::from(tour[0]), last_c)] -= 1;
        self.0[(last_c, usize::from(tour[0]))] -= 1;
    }
}

impl Index<(CityIndex, CityIndex)> for PathUsageMatrix {
    type Output = u16;

    fn index(&self, (x, y): (CityIndex, CityIndex)) -> &Self::Output {
        self.0.index((usize::from(x), usize::from(y)))
    }
}
