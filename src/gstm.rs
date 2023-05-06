//! Greedy sub-tour mutation algorithms as used in q[C]ABC
//! Defined in "Development a new mutation operator to solve the Traveling Salesman
//! Problem by aid of Genetic Algorithms"

use rand::{distributions::Uniform, Rng};

use crate::{
    config::{Float, MainRng},
    distance_matrix::DistanceMatrix,
    tour::{Tour, TourIndex},
};

// Generate neighbours from this tour in place (by mutating the passed-in tour).
// A new tour V_i is constructed from T_i taking cities from T_k.
pub fn generate_neighbour<R: Rng>(
    t_i: &Tour,
    t_k: &Tour,
    rng: &mut R,
    p_rc: Float,
    p_cp: Float,
    p_l: Float,
    nl_max: u16,
    distances: &DistanceMatrix,
) -> Tour {
    // Randomly select a city j.
    let j = TourIndex::random(rng, t_i.number_of_cities());
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

    let r1 = t_i.next_city(before_t_star_i);
    let r2 = t_i.previous_city(after_t_star_i);

    if rng.gen::<Float>() <= p_rc {
        // Add T^*_i to T^#_i so that a minimum extension is generated.
        t_i.reinsert_segment_for_maximum_gain(before_t_star_i, after_t_star_i, distances);
    } else {
        let mut t_hash_i = t_i.subtour_inclusive(after_t_star_i, before_t_star_i);
        let r1_idx = t_i.next_idx(before_t_star_i);
        let r1 = t_i[r1_idx];
        let r2_idx = t_i.previous_idx(after_t_star_i);
        let r2 = t_i[r2_idx];
        if rng.gen::<Float>() <= p_cp {
            t_i.distort(
                t_hash_i,
                r1_idx,
                r2_idx,
                Uniform::new(0.0, 1.0).unwrap(),
                rng,
                p_l,
                distances,
            );
        } else {
            let r1_neighbourhood_list = distances.neighbourhood_list(r1, nl_max);
            let r2_neighbourhood_list = distances.neighbourhood_list(r2, nl_max);
        }
    }

    todo!()
}
