use rand::{
    distributions::{Uniform, WeightedIndex},
    prelude::Distribution,
    seq::SliceRandom,
    Rng,
};

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
#[derive(Debug, Clone, Copy)]
pub struct Bee {}

impl Bee {
    pub fn new() -> Self {
        Self {}
    }

    // Returns a new tour if it is better than x_i.
    pub fn employed_phase<R: Rng>(
        self,
        x_i: &mut TourExt,
        t_k: &Tour,
        rng: &mut R,
        distances: &DistanceMatrix,
        neighbourhood_lists: &NeighbourMatrix,
        p_rc: Float,
        p_cp: Float,
        p_l: Float,
    ) {
        let v_i = gstm::generate_neighbour(
            x_i.tour(),
            t_k,
            rng,
            p_rc,
            p_cp,
            p_l,
            neighbourhood_lists,
            distances,
        );
        if v_i.is_shorter_than(x_i.tour()) {
            x_i.set_tour(v_i);
        }
    }

    pub fn onlooker_phase<R: Rng>(
        self,
        tours: &mut [TourExt],
        distrib: &WeightedIndex<Float>,
        rng: &mut R,
        distances: &DistanceMatrix,
        neighbourhood_lists: &NeighbourMatrix,
        p_rc: Float,
        p_cp: Float,
        p_l: Float,
    ) {
        let t_idx = distrib.sample(rng);
        let t = &tours[t_idx];

        let v_i = gstm::generate_neighbour(
            t.tour(),
            tours.choose(rng).unwrap().tour(),
            rng,
            p_rc,
            p_cp,
            p_l,
            neighbourhood_lists,
            distances,
        );
        if v_i.is_shorter_than(t.tour()) {
            tours[t_idx].set_tour(v_i);
        }
    }

    pub fn scout_phase<R: Rng>(
        self,
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
