use std::ops::Index;

use crate::{
    config::{self, Float},
    distance_matrix::DistanceMatrix,
    matrix::SquareMatrix,
    tour::CityIndex,
    tour::Tour,
    utils::order,
};

/// Upper right triangle: pheromone level.
/// Lower left triangle: visility.powf(beta).
pub struct PheromoneVisibilityMatrix {
    matrix: SquareMatrix<Float>,
    ro: Float,
}

impl PheromoneVisibilityMatrix {
    pub fn new(
        side_length: usize,
        init_value: Float,
        distances: &DistanceMatrix,
        beta: Float,
        ro: Float,
    ) -> PheromoneVisibilityMatrix {
        let mut matrix = SquareMatrix::new(side_length, init_value);

        // Pheromone level is already set when SquareMatrix is initialized.
        // Set lower left triangle: visibility.powf(beta).
        for y in 0..side_length {
            for x in 0..y {
                // We will use beta to raise the d_ij, not 1/d_ij,
                // so it must be negative to get the same results.
                matrix[(x, y)] = (distances[(x, y)] as Float).powf(-beta);
            }
        }
        PheromoneVisibilityMatrix { matrix, ro }
    }

    pub fn side_length(&self) -> usize {
        self.matrix.side_length()
    }

    // x must be higher than y
    pub fn pheromone(&self, (x, y): (CityIndex, CityIndex)) -> Float {
        debug_assert!(x > y);

        self.matrix[(x.get(), y.get())]
    }

    // x must be lower than y
    pub fn visibility(&self, (x, y): (CityIndex, CityIndex)) -> Float {
        debug_assert!(x < y);

        self.matrix[(x.get(), y.get())]
    }

    pub fn adjust_pheromone(&mut self, (x, y): (CityIndex, CityIndex), delta_tau: Float) {
        debug_assert!(x > y);

        self.matrix[(x.get(), y.get())] += delta_tau;
    }

    pub fn evaporate_pheromone(&mut self) {
        for x in 0..self.matrix.side_length() {
            for y in 0..x {
                self.matrix[(x, y)] *= self.ro;
            }
        }
    }
}
