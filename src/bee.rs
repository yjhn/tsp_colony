use rand::{seq::SliceRandom, Rng};

use crate::{
    config::Float,
    distance_matrix::DistanceMatrix,
    gstm,
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
        tours: &[(Tour, u32)],
        rng: &mut R,
        distances: &DistanceMatrix,
        p_rc: Float,
        p_cp: Float,
        p_l: Float,
        nl_max: u16,
    ) -> Option<Tour> {
        let t_i = x_i;
        let t_k = &tours.choose(rng).unwrap().0;
        let v_i = gstm::generate_neighbour(x_i, t_k, rng, p_rc, p_cp, p_l, nl_max, distances);
        if v_i.is_shorter_than(t_i) {
            Some(v_i)
        } else {
            None
        }
    }

    pub fn scout_phase<R: Rng>(
        (x_i, non_improvement_gens): &mut (Tour, u32),
        max_non_improvement_gens: u32,
        rng: &mut R,
        distances: &DistanceMatrix,
    ) {
        if *non_improvement_gens == max_non_improvement_gens {
            *x_i = Tour::random(x_i.number_of_cities() as u16, distances, rng);
            *non_improvement_gens = 0;
        }
    }
}
