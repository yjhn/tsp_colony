use rand::{seq::SliceRandom, Rng};

use crate::{
    cabc::{NeighbourMatrix, TourExt},
    config::{DistanceT, Float},
    distance_matrix::DistanceMatrix,
    gstm,
    index::CityIndex,
    matrix::Matrix,
    tour::{Tour, TourFunctions},
};

// Bees will not know which kind they are, they will
// expose methods for all bee types.
pub struct Bee {}

impl Bee {
    pub fn new() -> Self {
        Self {}
    }

    // Returns a new tour if it is better than x_i.
    pub fn employed_phase<R: Rng>(
        x_i: &Tour,
        tours: &[TourExt],
        rng: &mut R,
        distances: &DistanceMatrix,
        neighbourhood_lists: &NeighbourMatrix,
        p_rc: Float,
        p_cp: Float,
        p_l: Float,
    ) -> Option<Tour> {
        let t_k = tours.choose(rng).unwrap().tour();
        let v_i = gstm::generate_neighbour(
            x_i,
            t_k,
            rng,
            p_rc,
            p_cp,
            p_l,
            neighbourhood_lists,
            distances,
        );
        if v_i.is_shorter_than(x_i) {
            Some(v_i)
        } else {
            None
        }
    }

    pub fn scout_phase<R: Rng>(
        x_i: &mut TourExt,
        max_non_improvement_iters: u32,
        rng: &mut R,
        distances: &DistanceMatrix,
    ) {
        if x_i.non_improvement_iters() == max_non_improvement_iters {
            x_i.set_tour(Tour::random(
                x_i.tour().number_of_cities() as u16,
                distances,
                rng,
            ));
        }
    }
}
