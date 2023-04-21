use std::ops::Index;

use crate::{
    config, distance_matrix::DistanceMatrix, matrix::SquareMatrix, tour::CityIndex, tour::Tour,
    utils::order,
};

type PheromoneAmountT = f32;

/// Upper right triangle: pheromone level.
/// Lower left triangle: visility.powf(beta).
pub struct PheromoneVisibilityMatrix(SquareMatrix<PheromoneAmountT>);

impl PheromoneVisibilityMatrix {
    pub fn new(
        side_length: usize,
        init_value: PheromoneAmountT,
        distances: &DistanceMatrix,
        beta: f32,
    ) -> PheromoneVisibilityMatrix {
        let mut matrix = SquareMatrix::new(side_length, init_value);

        // Pheromone level is already set when SquareMatrix is initialized.
        // Set lower left triangle: visibility.powf(beta).
        for y in 0..side_length {
            for x in 0..y {
                // We will use beta to raise the d_ij, not 1/d_ij,
                // so it must be negative to get the same results.
                matrix[(x, y)] = (distances[(x, y)] as PheromoneAmountT).powf(-beta);
            }
        }
        PheromoneVisibilityMatrix(matrix)
    }

    pub fn side_length(&self) -> usize {
        self.0.side_length()
    }

    pub fn decrease_tour_pheromone(&mut self, t: &Tour) {
        for path in t.paths() {
            if let [c1, c2] = *path {
                self.decrease_prob(c1, c2);
            }
        }
        // Don't forget the last path.
        let (c1, c2) = t.last_to_first_path();
        self.decrease_prob(c1, c2);
    }

    fn decrease_prob(&mut self, c1: CityIndex, c2: CityIndex) {
        let (l, h) = order(c1.get(), c2.get());
        todo!()
        // let val = self.0[(h, l)] - todo!();
        // All values in probability matrix must always be in range [0..1].
        // self.0[(h, l)] = f64::clamp(val, 0.0, 1.0)
    }

    pub fn increase_tour_pheromone(&mut self, t: &Tour) {
        for path in t.paths() {
            if let [c1, c2] = *path {
                self.increase_prob(c1, c2);
            }
        }
        // Don't forget the last path.
        let (c1, c2) = t.last_to_first_path();
        self.increase_prob(c1, c2);
    }

    fn increase_prob(&mut self, c1: CityIndex, c2: CityIndex) {
        let (l, h) = order(c1.get(), c2.get());
        todo!()
        // let val = self.0[(h, l)] + todo!();
        // All values in probability matrix must always be in range [0..1].
        // self.0[(h, l)] = f64::clamp(val, 0.0, 1.0)
    }

    // x must be higher than y
    pub fn get_pheromone(&self, (x, y): (CityIndex, CityIndex)) -> PheromoneAmountT {
        debug_assert!(x > y);

        self.0[(x.get(), y.get())]
    }

    // x must be lower than y
    pub fn get_visibility(&self, (x, y): (CityIndex, CityIndex)) -> PheromoneAmountT {
        debug_assert!(x < y);

        self.0[(x.get(), y.get())]
    }
}
