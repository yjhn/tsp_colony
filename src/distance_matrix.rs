use crate::{config::Zeroable, index::CityIndex};
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

    // This is expensive, cache whenever possible.
    /// Returns `size` nearest neighbours of `city` in the form of (city_index, dist).
    pub fn neighbourhood_list(&self, city: CityIndex, size: u16) -> Vec<(CityIndex, DistanceT)> {
        // TODO: maybe there is a faster algorithm?
        // The algorithm: attach city numbers to each distance in `city` row of the
        // table. Then sort the table by distance, skip first (0 distance to self)
        // and take the required amount.
        let city = usize::from(city);
        let mut nl: Vec<(CityIndex, DistanceT)> = self
            .0
            .row(city)
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, dist)| (CityIndex::new(idx as u16), dist))
            .collect();
        // Remove distnce to self.
        nl.swap_remove(city);
        nl.sort_unstable_by_key(|&(idx, dist)| dist);
        // Leave the required number of items.
        nl.truncate(usize::from(size));

        nl
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
