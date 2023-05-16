use crate::{config::Zeroable, index::CityIndex, matrix::Matrix};
use std::ops::Index;

use crate::{
    config::{DistanceT, Float},
    matrix::SquareMatrix,
};

/// Distances are stored in the upper right corner.
#[derive(Debug, Clone)]
pub struct DistanceMatrix(SquareMatrix<DistanceT>);

impl DistanceMatrix {
    pub fn new(side_length: usize) -> DistanceMatrix {
        DistanceMatrix(SquareMatrix::new(side_length, DistanceT::ZERO))
    }

    pub fn side_length(&self) -> usize {
        self.0.side_length()
    }

    pub fn set_dist(&mut self, c1: CityIndex, c2: CityIndex, distance: DistanceT) {
        // Distance matrix is symmetric.
        self.0[(c1.into(), c2.into())] = distance;
        self.0[(c2.into(), c1.into())] = distance;
    }

    /// Constructs a neighbourhood lists matrix. It will have `city_count` rows
    /// and `size` columns.
    pub fn neighbourhood_lists(&self, size: u16) -> Matrix<(CityIndex, DistanceT)> {
        let mut matrix = Matrix::new(
            usize::from(size),
            self.side_length(),
            (CityIndex::new(0), DistanceT::ZERO),
        );

        for city_idx in 0..self.side_length() {
            let row = matrix.row_mut(city_idx);

            // TODO: maybe there is a faster algorithm?
            // The algorithm: attach city numbers to each distance in `city` row of the
            // table. Then sort the row by distance, skip first (0 distance to self)
            // and take the required amount.
            let city = city_idx;
            let mut neighbours: Vec<(CityIndex, DistanceT)> = self
                .0
                .row(city)
                .iter()
                .copied()
                .enumerate()
                .map(|(idx, dist)| (CityIndex::new(idx as u16), dist))
                .collect();
            neighbours.sort_unstable_by_key(|&(_idx, dist)| dist);
            // Copy neighbours to slice.
            for (elem, place_in_row) in neighbours
                .iter()
                .copied()
                .skip(1)
                .take(usize::from(size + 1))
                .zip(row)
            {
                // After we have skipped self, indexes will be off by +1.
                *place_in_row = elem;
            }
        }

        matrix
    }
}

// First number must be higher than second.
impl Index<(usize, usize)> for DistanceMatrix {
    type Output = DistanceT;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        &self.0[(x, y)]
    }
}

impl Index<(CityIndex, CityIndex)> for DistanceMatrix {
    type Output = DistanceT;

    fn index(&self, (x, y): (CityIndex, CityIndex)) -> &Self::Output {
        &self.0[(x.into(), y.into())]
    }
}
