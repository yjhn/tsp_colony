//! Greedy sub-tour mutation algorithms as used in q[C]ABC

use rand::Rng;

use crate::{
    config::{Float, MainRng},
    distance_matrix::DistanceMatrix,
    tour::Tour,
};

// Generate neighbours from this tour in place (by mutating the passed-in tour).
// A new tour V_i is constructed from T_i taking cities from T_k.
pub fn generate_neighbours<R: Rng>(
    t_i: &Tour,
    t_k: &Tour,
    rng: &mut R,
    p_rc: Float,
    p_cp: Float,
    distances: &DistanceMatrix,
) {
    // Randomly select a city j.
    let j = rng.gen_range(0..t_i.number_of_cities());
    let j_k = t_k.cities()[j];
    let j_k_pos_in_t_i = t_i.index_of(j_k);
    // Randomly select direction value.
    let fi: bool = rng.gen();
    // T* is the open tour segment.
    let (before_t_star_i, after_t_star_i) = if fi {
        // Forward direction.
        let after_j_k = t_k.next_city(j);
        let after_j_k_pos_in_t_i = t_i.index_of(after_j_k);
        // let t_hash_i = t_i.subtour(j_k_pos_in_t_i, after_j_k_pos_in_t_i);
        (j_k_pos_in_t_i, after_j_k_pos_in_t_i)
    } else {
        // Backward direction.
        let before_j_k = t_k.previous_city(j);
        let before_j_k_pos_in_t_i = t_i.index_of(before_j_k);
        // Cities in t_i between before_j_k_pos_in_t_i and before_j_k form an unclosed
        // subtour T^#_i
        // let t_hash_i = t_i.subtour(before_j_k_pos_in_t_i, j_k_pos_in_t_i);
        (before_j_k_pos_in_t_i, j_k_pos_in_t_i)
    };

    // TODO: cloning this is not neccessary, we can just look at T_i.
    // let mut t_star_i = t_i.subtour(before_t_star_i, after_t_star_i);
    // let r1 = t_star_i[0];
    // let r2 = *t_star_i.last().unwrap();

    let mut t_hash_i = t_i.subtour_inclusive(after_t_star_i, before_t_star_i);
    // TODO: it would maybe be more efficient to simply reverse / not reverse the segment.
    if rng.gen::<Float>() <= p_rc {
        // TODO: we need to determine this by taking into account ALL cites in T*, not just the place wher it was before!!!
        if distances[(r1, t_i.cities()[before_t_star_i])]
            + distances[(r2, t_i.cities()[after_t_star_i])]
            < distances[(r1, t_i.cities()[after_t_star_i])]
                + distances[(r2, t_i.cities()[before_t_star_i])]
        {
            // insert after_t_hash_i...before_t_hash_i at the end
        } else {
            // insert same, but reversed
        }
    } else {
        if rng.gen::<Float>() <= p_cp {
            let r1_pos_in_t_i = t_i.next_idx(before_t_star_i);
            let r2_pos_in_t_i = t_i.previous_idx(after_t_star_i);
            // TODO: add cities from T_i to t_hash_i "by rolling or mixing". Also delete
        } else {
        }
    }
}
