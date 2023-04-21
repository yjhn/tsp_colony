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
}

// First number must be higher than second.
impl Index<(usize, usize)> for DistanceMatrix {
    type Output = DistanceT;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        debug_assert!(x > y);

        &self.0[(x, y)]
    }
}

// First number must be higher than second.
impl Index<(CityIndex, CityIndex)> for DistanceMatrix {
    type Output = DistanceT;

    fn index(&self, (x, y): (CityIndex, CityIndex)) -> &Self::Output {
        debug_assert!(x > y);

        &self.0[(x.get(), y.get())]
    }
}
// First number must be higher than second.
impl IndexMut<(CityIndex, CityIndex)> for DistanceMatrix {
    fn index_mut(&mut self, (x, y): (CityIndex, CityIndex)) -> &mut Self::Output {
        debug_assert!(x > y);

        &mut self.0[(x.get(), y.get())]
    }
}
