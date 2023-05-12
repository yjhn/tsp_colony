//! Greedy sub-tour mutation algorithm as used in q\[C\]ABC
//! Defined in "Development a new mutation operator to solve the Traveling Salesman
//! Problem by aid of Genetic Algorithms"

use rand::{distributions::Uniform, prelude::Distribution, seq::SliceRandom, Rng};

use crate::{
    config::{DistanceT, Float, MainRng},
    distance_matrix::DistanceMatrix,
    index::{CityIndex, TourIndex},
    matrix::Matrix,
    qcabc::NeighbourMatrix,
    tour::{Tour, TourFunctions},
};

// Generate neighbour from this tour.
// A new tour V_i is constructed from T_i taking cities from T_k.
pub fn generate_neighbour<R: Rng>(
    t_i: &Tour,
    t_k: &Tour,
    p_rc: Float,
    p_cp: Float,
    p_l: Float,
    min_subtour_cities: usize,
    max_subtour_cities: usize,
    neighbourhood_lists: &NeighbourMatrix,
    distances: &DistanceMatrix,
    cities_distrib: Uniform<u16>,
    distrib01: Uniform<Float>,
    rng: &mut R,
) -> Tour {
    // t_star_i is the open tour segment.
    let (before_t_star_i, after_t_star_i) = select_subtour(
        t_i,
        t_k,
        min_subtour_cities,
        max_subtour_cities,
        cities_distrib,
        rng,
    );
    debug_assert_eq!(t_i.length(), t_i.calculate_tour_length(distances));

    let r1 = t_i.next_city(before_t_star_i);
    let r2 = t_i.previous_city(after_t_star_i);

    if distrib01.sample(rng) <= p_rc {
        // Add T^*_i to T^#_i so that a minimum extension is generated.
        let new = t_i.reinsert_segment_for_maximum_gain(before_t_star_i, after_t_star_i, distances);
        debug_assert_eq!(new.length(), new.calculate_tour_length(distances));
        new
    } else {
        let r1_idx = t_i.next_idx(before_t_star_i);
        let r2_idx = t_i.previous_idx(after_t_star_i);
        if distrib01.sample(rng) <= p_cp {
            let t_hash_i = t_i.subtour_inclusive(after_t_star_i, before_t_star_i);
            t_i.distort(t_hash_i, r1_idx, r2_idx, distrib01, rng, p_l, distances)
        } else {
            let r1_neighbourhood_list = neighbourhood_lists.row(usize::from(r1));
            let r2_neighbourhood_list = neighbourhood_lists.row(usize::from(r2));
            // TODO: kaip suprantu, paliekam t_i nepakeistą, tik nl_r1 arba nl_r2 perkeliam kitur?
            let mut new_tour = t_i.clone();
            // Choose random neighbours.
            let (nl_r1, dist_r1, nl_r1_in_tour) = choose_not_immediately_preceding(
                r1_neighbourhood_list,
                rng,
                r1_idx,
                new_tour.cities(),
            );
            let (nl_r2, dist_r2, nl_r2_in_tour) = choose_not_immediately_preceding(
                r2_neighbourhood_list,
                rng,
                r2_idx,
                new_tour.cities(),
            );
            let r1_prec = new_tour[before_t_star_i];
            let nl_r1_prec = new_tour.previous_idx(nl_r1_in_tour);
            let r2_prec_idx = new_tour.previous_idx(r2_idx);
            let r2_prec = new_tour[r2_prec_idx];
            let nl_r2_prec = new_tour.previous_idx(nl_r2_in_tour);
            let gain_r1 =
                Tour::gain_from_2_opt(new_tour[nl_r1_prec], nl_r1, r1_prec, r1, dist_r1, distances);
            let gain_r2 =
                Tour::gain_from_2_opt(new_tour[nl_r2_prec], nl_r2, r2_prec, r2, dist_r2, distances);
            // TODO: we don't need to reverse segment in already contructed tour; we can
            // simply construct the tour with the segment already reversed.
            if gain_r1 > gain_r2 {
                new_tour.reverse_segment(nl_r1_in_tour, before_t_star_i, gain_r1);
            } else {
                new_tour.reverse_segment(nl_r2_in_tour, r2_prec_idx, gain_r2);
            }
            debug_assert_eq!(new_tour.length(), new_tour.calculate_tour_length(distances));
            new_tour
        }
    }
}

// Original GSTM paper says "not a neighbour", qCABC paper says "not preceding".
// I implemented qCABC, so chose "not preceding".
fn choose_not_immediately_preceding<R: Rng>(
    neighbourhood_list: &[(CityIndex, DistanceT)],
    rng: &mut R,
    relative_to: TourIndex,
    tour: &[CityIndex],
) -> (CityIndex, DistanceT, TourIndex) {
    loop {
        let &(neigh, dist_neigh) = neighbourhood_list.choose(rng).unwrap();
        let neigh_in_tour = tour.index_of(neigh);
        debug_assert_ne!(neigh_in_tour, relative_to);
        if neigh_in_tour > relative_to || relative_to - neigh_in_tour > 1 {
            return (neigh, dist_neigh, neigh_in_tour);
        }
    }
}

/// Returns (`before_t_star_i`, `after_t_star_i`).
fn select_subtour<R: Rng>(
    t_i: &[CityIndex],
    t_k: &[CityIndex],
    min_subtour_cities: usize,
    max_subtour_cities: usize,
    cities_distrib: Uniform<u16>,
    rng: &mut R,
) -> (TourIndex, TourIndex) {
    loop {
        // Randomly select a city j.
        let j = TourIndex::random(cities_distrib, rng);
        let j_k = t_k[j];
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

        // Number of cities in segment must be between `min_subtour_cities` and `max_subtour_cities`.
        debug_assert!(t_i.segment_len(before_t_star_i, after_t_star_i) >= 2);
        let segment_len = t_i.segment_len(before_t_star_i, after_t_star_i) - 2;
        if (min_subtour_cities..=max_subtour_cities).contains(&segment_len) {
            return (before_t_star_i, after_t_star_i);
        }
    }
}
