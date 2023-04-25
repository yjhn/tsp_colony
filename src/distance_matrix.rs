use std::ops::{Index, IndexMut};

use crate::{config, matrix::SquareMatrix, tour::CityIndex, tour::Tour, utils::order};

type DistanceT = u32;

/// Distances are stored in the upper right corner.
#[derive(Debug, Clone)]
pub struct DistanceMatrix(SquareMatrix<DistanceT>);

impl DistanceMatrix {
    pub fn new(side_length: usize) -> DistanceMatrix {
        DistanceMatrix(SquareMatrix::new(side_length, 0))
    }

    pub fn side_length(&self) -> usize {
        self.0.side_length()
    }

    pub fn set_dist(&mut self, c1: CityIndex, c2: CityIndex, distance: DistanceT) {
        // Distance matrix is symmetric.
        self.0[(c1.get(), c2.get())] = distance;
        self.0[(c2.get(), c1.get())] = distance;
    }
}

// First number must be higher than second.
impl Index<(usize, usize)> for DistanceMatrix {
    type Output = DistanceT;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        debug_assert!(x > y);

        &self.0[(x, y)]
    }
}

impl Index<(CityIndex, CityIndex)> for DistanceMatrix {
    type Output = DistanceT;

    fn index(&self, (x, y): (CityIndex, CityIndex)) -> &Self::Output {
        &self.0[(x.get(), y.get())]
    }
}
