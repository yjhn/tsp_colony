use crate::config::DistanceT;
use crate::config::Float;
use crate::config::Zeroable;
use crate::distance_matrix::DistanceMatrix;
use crate::matrix::SquareMatrix;
use crate::static_assert;
use crate::utils::all_cities;
use crate::utils::order;
use crate::utils::reverse_order;
use mpi::traits::Equivalence;
use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::prelude::SliceRandom;
use rand::Rng;
use std::fmt::write;
use std::fmt::Debug;
use std::fmt::Display;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::mem::size_of;
use std::ops::Index;
use std::ops::Sub;
use std::path::Path;
use std::slice::Windows;

/// Index of the city in the city matrix.
/// `u16` is enough since it is extremely unlikely that number of cities would be greater.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Equivalence)]
pub struct CityIndex(u16);

impl CityIndex {
    pub fn new(index: u16) -> CityIndex {
        CityIndex(index)
    }

    pub fn from_be_bytes(b1: u8, b2: u8) -> CityIndex {
        CityIndex::new(u16::from_be_bytes([b1, b2]))
    }
}

impl From<CityIndex> for u16 {
    fn from(value: CityIndex) -> Self {
        value.0
    }
}

impl From<CityIndex> for usize {
    fn from(value: CityIndex) -> Self {
        value.0.into()
    }
}

impl From<&CityIndex> for usize {
    fn from(value: &CityIndex) -> Self {
        value.0.into()
    }
}

impl Debug for CityIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "c{}", self.0)
    }
}

impl Display for CityIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "CityIndex({})", self.0)
        write!(f, "c{}", self.0)
    }
}

/// Type for tour indexing.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TourIndex(usize);

impl TourIndex {
    pub fn new(idx: usize) -> Self {
        Self(idx)
    }

    pub fn random<R: Rng>(rng: &mut R, number_of_cities: usize) -> Self {
        Self::new(rng.gen_range(0..number_of_cities))
    }

    pub fn wrapping_inc(self, max: usize) -> Self {
        if self.0 == max {
            Self::new(0)
        } else {
            Self::new(self.0 + 1)
        }
    }

    pub fn wrapping_dec(self, max: usize) -> Self {
        if self.0 > 0 {
            Self::new(self.0 - 1)
        } else {
            Self::new(max)
        }
    }

    pub fn is_last(self, max: usize) -> bool {
        self.0 == max
    }

    pub fn is_first(self) -> bool {
        self.0 == 0
    }
}

impl Display for TourIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "t{}", self.0)
    }
}

impl Debug for TourIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "t{}", self.0)
    }
}

impl From<TourIndex> for usize {
    fn from(value: TourIndex) -> Self {
        value.0
    }
}

impl Sub<TourIndex> for TourIndex {
    type Output = usize;

    fn sub(self, rhs: TourIndex) -> Self::Output {
        self.0 - rhs.0
    }
}

#[derive(Debug, Clone)]
pub struct Tour {
    cities: Vec<CityIndex>,
    tour_length: DistanceT,
}

// This disallows using f64 for Float.
static_assert!(size_of::<Float>() == 2 * size_of::<CityIndex>());

impl Tour {
    // How many elements are appended using hack methods for tour exchange.
    // We append delta_tau and tour length.
    // pub const APPENDED_HACK_ELEMENTS: usize = size_of::<Float>() / 2 + 2;
    pub const APPENDED_HACK_ELEMENTS: usize =
        size_of::<u32>() / size_of::<CityIndex>() + size_of::<Float>() / size_of::<CityIndex>();

    pub const PLACEHOLDER: Tour = Tour {
        cities: Vec::new(),
        tour_length: DistanceT::MAX,
    };

    pub fn number_of_cities(&self) -> usize {
        self.cities.len()
    }

    pub fn length(&self) -> DistanceT {
        self.tour_length
    }

    pub fn cities(&self) -> &[CityIndex] {
        &self.cities
    }

    pub fn index_of(&self, city: CityIndex) -> TourIndex {
        TourIndex::new(self.cities.iter().position(|&elem| elem == city).unwrap())
    }

    pub fn from_cities(cities: Vec<CityIndex>, distances: &DistanceMatrix) -> Tour {
        let tour_length = cities.calculate_tour_length(distances);

        Tour {
            cities,
            tour_length,
        }
    }

    // pub fn from_hack_cities(mut cities_with_length: Vec<CityIndex>) -> Tour {
    //     // Length is stored as two CityIndex'es in big endian order (i. e. u16 at [end]
    //     // is the lower end and u16 at [end - 1] is the higher end)
    //     let tour_length_part_1: u16 = cities_with_length.pop().unwrap().into();
    //     let tour_length_part_2: u16 = cities_with_length.pop().unwrap().into();
    //     let [b3, b4] = tour_length_part_1.to_be_bytes();
    //     let [b1, b2] = tour_length_part_2.to_be_bytes();
    //     let tour_length = u32::from_be_bytes([b1, b2, b3, b4]);
    //     // Cities don't contain length anymore.
    //     let cities = cities_with_length;

    //     Tour {
    //         cities,
    //         tour_length,
    //     }
    // }

    pub fn random(city_count: u16, distances: &DistanceMatrix, rng: &mut impl Rng) -> Tour {
        assert!(city_count > 1);

        let mut cities = all_cities(city_count);
        cities.shuffle(rng);
        let tour_length = cities.calculate_tour_length(distances);

        Tour {
            cities,
            tour_length,
        }
    }

    pub fn is_shorter_than(&self, other: &Tour) -> bool {
        self.tour_length < other.tour_length
    }

    pub fn save_to_file<P: AsRef<Path>, Dp: AsRef<Path> + Display>(
        &self,
        problem_path: Dp,
        path: P,
    ) {
        let file = File::create(path).unwrap();
        let mut file = BufWriter::new(file);

        writeln!(file, "Problem file: {problem_path}").unwrap();
        writeln!(file, "Number of cities: {}", self.number_of_cities()).unwrap();
        writeln!(file, "Tour length: {}", self.tour_length).unwrap();
        writeln!(file, "Cities:").unwrap();

        // Use indices starting at 1 for output, same as TSPLIB
        // format for consistency.
        for city in self.cities.iter().copied() {
            write!(file, "{} ", u16::from(city) + 1).unwrap();
        }
        writeln!(file).unwrap();
    }

    pub fn last_to_first_path(&self) -> (CityIndex, CityIndex) {
        (*self.cities.last().unwrap(), self.cities[0])
    }

    // Returns iterator over all paths except for the
    // first -> last.
    pub fn paths(&self) -> Windows<'_, CityIndex> {
        self.cities.windows(2)
    }

    /// Creates a new `Tour` by cloning the provided slice. Trusts that `length`
    /// is correct.
    pub fn clone_from_cities(
        tour: &[CityIndex],
        length: DistanceT,
        distances: &DistanceMatrix,
    ) -> Tour {
        debug_assert_eq!(length, tour.calculate_tour_length(distances));

        Tour {
            cities: tour.to_owned(),
            tour_length: length,
        }
    }

    pub fn distance(&self, other: &[CityIndex]) -> u16 {
        self.cities.distance(other)
    }

    /// Returns true if the tour being constructed by this ant has in edge between `first`
    /// and `second`. Order does not matter.
    pub fn has_path(&self, x: CityIndex, y: CityIndex) -> bool {
        self.cities.has_path(x, y)
    }

    // This function call must be matched by the corresponding
    // call to remove_hack_length_and_mpi_rank().
    // pub fn hack_append_length_and_mpi_rank(&mut self, rank: i32) {
    pub fn hack_append_length_cvg(&mut self, cvg: Float) {
        // Tour length.
        let [b1, b2, b3, b4] = self.tour_length.to_be_bytes();
        self.cities.push(CityIndex::from_be_bytes(b1, b2));
        self.cities.push(CityIndex::from_be_bytes(b3, b4));

        let [cb1, cb2, cb3, cb4] = cvg.to_be_bytes();
        self.cities.push(CityIndex::from_be_bytes(cb1, cb2));
        self.cities.push(CityIndex::from_be_bytes(cb3, cb4));
        // MPI rank.
        // let [b1, b2, b3, b4] = rank.to_be_bytes();
        // self.cities
        //     .push(CityIndex::new(u16::from_be_bytes([b1, b2])));
        // self.cities
        //     .push(CityIndex::new(u16::from_be_bytes([b3, b4])));
    }

    // pub fn remove_hack_length_and_mpi_rank(&mut self) {
    pub fn remove_hack_length_cvg(&mut self) {
        for _ in 0..Self::APPENDED_HACK_ELEMENTS {
            self.cities.pop();
        }
    }

    pub fn swap(&mut self, city_1: TourIndex, city_2: TourIndex) {
        self.cities.swap(city_1.into(), city_2.into());
    }

    // Tour is circular.
    pub fn previous_idx(&self, city: TourIndex) -> TourIndex {
        debug_assert!(usize::from(city) < self.number_of_cities());
        city.wrapping_dec(self.number_of_cities() - 1)
    }

    pub fn previous_city(&self, city: TourIndex) -> CityIndex {
        // Implementing Index for slice does not make it work for Vec, only on slice.
        self.cities.as_slice()[self.previous_idx(city)]
    }

    pub fn next_idx(&self, city: TourIndex) -> TourIndex {
        debug_assert!(usize::from(city) < self.number_of_cities());
        city.wrapping_inc(self.number_of_cities() - 1)
    }

    pub fn next_city(&self, city: TourIndex) -> CityIndex {
        self.cities.as_slice()[self.next_idx(city)]
    }

    /// Returns tour segment between (not including) `first` and `last` in tour order
    /// as the first elements in a newly created [Vec].
    pub fn subtour(&self, first: TourIndex, last: TourIndex) -> Vec<CityIndex> {
        let (first, last) = (usize::from(first), usize::from(last));
        debug_assert_ne!(first, last);
        debug_assert!(first < self.number_of_cities() && last < self.number_of_cities());
        let mut segment = Vec::with_capacity(self.number_of_cities());
        if first > last {
            // if `first` is the last index and `last` is 0, this returns empty segment
            // From `first` to the end...
            segment.extend_from_slice(&self.cities[(first + 1)..]);
            // ...and from start to `last`.
            segment.extend_from_slice(&self.cities[..last]);
        } else {
            // first < last
            // if first - last == 1, this returns empty segment
            segment.extend_from_slice(&self.cities[(first + 1)..last]);
        }
        segment
    }

    /// Returns tour segment between (and including) `first` and `last` in tour order
    /// as the first elements in a newly created [Vec].
    pub fn subtour_inclusive(&self, first: TourIndex, last: TourIndex) -> Vec<CityIndex> {
        let (first, last) = (usize::from(first), usize::from(last));
        debug_assert_ne!(first, last);
        debug_assert!(first < self.number_of_cities() && last < self.number_of_cities());
        let mut segment = Vec::with_capacity(self.number_of_cities());
        if first > last {
            // if `first` is the last index and `last` is 0, this returns empty segment
            // From `first` to the end...
            segment.extend_from_slice(&self.cities[first..]);
            // ...and from start to `last`.
            segment.extend_from_slice(&self.cities[..=last]);
        } else {
            // first < last
            // if first - last == 1, this returns empty segment
            segment.extend_from_slice(&self.cities[first..=last]);
        }
        segment
    }

    // Similar to `subtour`, but this appends and is inclusive.
    pub fn append_subtour(&self, first: usize, last: usize, buf: &mut Vec<CityIndex>) {
        debug_assert_ne!(first, last);
        debug_assert!(first < self.number_of_cities() && last < self.number_of_cities());
        if first > last {
            // From `first` to the end...
            buf.extend_from_slice(&self.cities[first..]);
            // ...and from start to `last`.
            buf.extend_from_slice(&self.cities[..=last]);
        } else {
            // first < last
            buf.extend_from_slice(&self.cities[first..=last]);
        }
    }

    pub fn append_reversed_subtour(&self, first: usize, last: usize, buf: &mut Vec<CityIndex>) {
        debug_assert_ne!(first, last);
        debug_assert!(first < self.number_of_cities() && last < self.number_of_cities());
        if first > last {
            // From `last` to start...
            buf.extend(self.cities[..=last].iter().rev());
            // ...and from last to `first`.
            buf.extend(self.cities[first..].iter().rev());
        } else {
            // first < last
            buf.extend(self.cities[first..=last].iter().rev());
        }
    }

    /// Number of cities in segment, `start` and `end` included.
    pub fn segment_len(&self, start: TourIndex, end: TourIndex) -> usize {
        if start <= end {
            end - start + 1
        } else {
            // Segment wraps around the end.
            (self.number_of_cities() - 1 - usize::from(start)) + usize::from(end) + 1
        }
    }

    fn update_shortest(
        current_shortest_before: &mut usize,
        current_shortest_reversed: &mut bool,
        current_shortest_added_paths_len: &mut DistanceT,
        (segment_start, segment_end): (CityIndex, CityIndex),
        idx: usize,
        (c1, c2): (CityIndex, CityIndex),
        distances: &DistanceMatrix,
    ) {
        let regular_extension = distances[(c1, segment_start)] + distances[(segment_end, c2)];
        let reversed_extension = distances[(c2, segment_start)] + distances[(segment_end, c1)];

        if regular_extension < reversed_extension {
            if regular_extension < *current_shortest_added_paths_len {
                *current_shortest_before = idx;
                *current_shortest_reversed = false;
                *current_shortest_added_paths_len = regular_extension;
            }
        } else if reversed_extension < *current_shortest_added_paths_len {
            *current_shortest_before = idx;
            *current_shortest_reversed = true;
            *current_shortest_added_paths_len = reversed_extension;
        }
    }

    /// Returns true if `c2` goes right after `c1` in the tour.
    pub fn is_next(&self, c1: TourIndex, c2: TourIndex) -> bool {
        if c1 < c2 {
            c2 - c1 == 1
        } else {
            c1.is_last(self.number_of_cities() - 1) && c2.is_first()
        }
    }

    // TODO: galima paminėti, kad čia visai panašu į 3-opt subset, kai jau pasirinkom
    // iškerpamą segmentą.
    /// Returns a new tour with segment beween (not including) `before_segment_start`
    /// and `after_segment_end` inserted into the location that minimizes the length.
    pub fn reinsert_segment_for_maximum_gain(
        &self,
        before_segment_start: TourIndex,
        after_segment_end: TourIndex,
        distances: &DistanceMatrix,
    ) -> Tour {
        // Resulting tour is identical to the existing one if ase is right after bss (then the
        // segment is empty) or bss == ase (the segment takes up all but one city).
        if self.is_next(before_segment_start, after_segment_end)
            || before_segment_start == after_segment_end
        {
            // TODO: maybe, to avoid unneeded work by the caller, return an Option<Tour>,
            // with this case being None?
            return self.clone();
        }

        let before_segment_start_c = self[before_segment_start];
        let after_segment_end_c = self[after_segment_end];
        let segment_start_idx = self.next_idx(before_segment_start);
        let segment_start = self[segment_start_idx];
        let segment_end_idx = self.previous_idx(after_segment_end);
        let segment_end = self[segment_end_idx];
        let removed_paths_length = distances[(before_segment_start_c, segment_start)]
            + distances[(segment_end, after_segment_end_c)];
        // City right before the segment insertion location.
        let mut new_segment_before = 0;
        // Whether the segment should be inserted reversed.
        let mut reversed = false;
        // Length difference to the current tour.
        let mut added_paths_length = DistanceT::MAX;
        // Choose where to insert the segment.
        if before_segment_start < after_segment_end {
            // Segment to be reinserted is in the middle.
            // Two parts: |-(first)- bss ----- (segment) ----- ase -(second)-|.
            for (idx, path) in self.cities[usize::from(after_segment_end)..]
                .windows(2)
                .enumerate()
            {
                let &[c1, c2] = path else {unreachable!()};
                Self::update_shortest(
                    &mut new_segment_before,
                    &mut reversed,
                    &mut added_paths_length,
                    (segment_start, segment_end),
                    idx + usize::from(after_segment_end),
                    (c1, c2),
                    distances,
                );
            }

            // Last to first city.
            Self::update_shortest(
                &mut new_segment_before,
                &mut reversed,
                &mut added_paths_length,
                (segment_start, segment_end),
                self.number_of_cities() - 1,
                (self.cities[0], *self.cities.last().unwrap()),
                distances,
            );

            // First to before_segment_start.
            for (idx, path) in self.cities[..=usize::from(before_segment_start)]
                .windows(2)
                .enumerate()
            {
                let &[c1, c2] = path else {unreachable!()};
                Self::update_shortest(
                    &mut new_segment_before,
                    &mut reversed,
                    &mut added_paths_length,
                    (segment_start, segment_end),
                    idx + usize::from(after_segment_end),
                    (c1, c2),
                    distances,
                );
            }
        } else {
            // Segment wraps around the tour end.
            for (idx, path) in self.cities
                [usize::from(after_segment_end)..=usize::from(before_segment_start)]
                .windows(2)
                .enumerate()
            {
                let &[c1,c2] = path else {unreachable!()};
                Self::update_shortest(
                    &mut new_segment_before,
                    &mut reversed,
                    &mut added_paths_length,
                    (segment_start, segment_end),
                    // Enumerate counts items from the beggining of the iterator.
                    idx + usize::from(before_segment_start),
                    (c1, c2),
                    distances,
                );
            }
        }

        // Construct a new Tour with the segment inserted into the chosen location.
        let mut cities_new = Vec::with_capacity(self.number_of_cities());

        let new_segment_after: usize = self.next_idx(TourIndex::new(new_segment_before)).into();
        // New segment will be between new_segment_before and new_segment_after.
        if new_segment_before > new_segment_after {
            // Non-segment.
            cities_new.extend_from_slice(&self.cities[new_segment_after..=new_segment_before]);
            // Segment is in two parts.
            if reversed {
                if new_segment_before == self.number_of_cities() - 1 {
                    cities_new.extend(self.cities[..new_segment_after].iter().rev());
                } else {
                    // Because the segment is reversed, we first add its reversed "end",
                    // then reversed "front".
                    cities_new.extend(self.cities[..new_segment_after].iter().rev());
                    cities_new.extend(self.cities[new_segment_before + 1..].iter().rev());
                }
            } else {
                if new_segment_before == self.number_of_cities() - 1 {
                    cities_new.extend_from_slice(&self.cities[..new_segment_after]);
                } else {
                    cities_new.extend_from_slice(&self.cities[new_segment_before + 1..]);
                    cities_new.extend_from_slice(&self.cities[..new_segment_after]);
                }
            }
        } else {
            // new_segment_before < new_segment_after
            // Non-segment is in two parts: ..=new_segment_before and new_segment_after..
            cities_new.extend_from_slice(&self.cities[..=new_segment_before]);

            if reversed {
                cities_new.extend(
                    self.cities[new_segment_before + 1..new_segment_after]
                        .iter()
                        .rev(),
                );
            } else {
                cities_new
                    .extend_from_slice(&self.cities[new_segment_before + 1..new_segment_after]);
            }

            cities_new.extend_from_slice(&self.cities[new_segment_after..]);
        }

        Tour {
            cities: cities_new,
            tour_length: self.tour_length - removed_paths_length + added_paths_length,
        }
    }

    // Since R1 is the last index of new_tour, we simply append cities to it.
    pub fn distort<R: Rng>(
        &self,
        mut t_hash: Vec<CityIndex>,
        r1_idx: TourIndex,
        r2_idx: TourIndex,
        distrib01: Uniform<Float>,
        rng: &mut R,
        p_l: Float,
        distances: &DistanceMatrix,
    ) -> Tour {
        // If we didn't create this subtour, we would need to keep track of used indices anyway,
        // which would probably be much slower.
        let mut t_star = self.subtour_inclusive(r1_idx, r2_idx);
        for _ in 0..t_star.len() {
            // TODO: use rng.gen_bool() where applicable.
            if distrib01.sample(rng) <= p_l {
                let chosen_city = rng.gen_range(0..t_star.len());
                t_hash.push(t_star[chosen_city]);
                t_star.swap_remove(chosen_city);
            } else {
                t_hash.push(t_star.pop().unwrap());
            }
        }

        let tour_length = t_hash.calculate_tour_length(distances);
        Tour {
            cities: t_hash,
            tour_length,
        }
    }
}

pub trait TourFunctions {
    fn calculate_tour_length(&self, distances: &DistanceMatrix) -> DistanceT;

    fn get_hack_tour_length(&self) -> DistanceT;

    fn get_hack_cvg(&self) -> Float;

    fn paths(&self) -> Windows<CityIndex>;

    fn has_path(&self, x: CityIndex, y: CityIndex) -> bool;

    fn distance(&self, other: &[CityIndex]) -> u16;

    fn update_pheromone(&self, delta_tau_matrix: &mut SquareMatrix<Float>, delta_tau: Float);
}

impl TourFunctions for [CityIndex] {
    fn calculate_tour_length(&self, distances: &DistanceMatrix) -> DistanceT {
        assert!(self.len() > 1);

        let mut tour_length = DistanceT::ZERO;
        for pair in self.paths() {
            tour_length += distances[(pair[0], pair[1])];
        }
        // Add distance from last to first.
        tour_length += distances[(*self.last().unwrap(), self[0])];

        tour_length
    }

    // The length must first be inserted using Tour::hack_append_length_and_mpi_rank().
    fn get_hack_tour_length(&self) -> DistanceT {
        // Length is stored as two CityIndex'es in big endian order (i. e. u16 at [end]
        // is the lower end and u16 at [end - 1] is the higher end)
        let tour_length_part_1: u16 = self[self.len() - 3].into();
        let tour_length_part_2: u16 = self[self.len() - 4].into();
        // let tour_length_part_1: u16 = self[self.len() - 1].into();
        // let tour_length_part_2: u16 = self[self.len() - 2].into();
        let [b3, b4] = tour_length_part_1.to_be_bytes();
        let [b1, b2] = tour_length_part_2.to_be_bytes();
        DistanceT::from_be_bytes([b1, b2, b3, b4])
    }

    fn has_path(&self, x: CityIndex, y: CityIndex) -> bool {
        let inner_paths = self.paths().any(|pair| {
            let &[c1, c2] = pair else { unreachable!() };
            order(x, y) == order(c1, c2)
        });
        inner_paths || order(x, y) == order(self[0], *self.last().unwrap())
    }

    /// Returns the number of paths (edges) in one tour that are not in other.
    /// Both tours must have the same number of cities.
    fn distance(&self, other: &[CityIndex]) -> u16 {
        debug_assert_eq!(self.len(), other.len());

        // City count and path count is the same.
        let num_cities = self.len() as u16;
        let mut common_edges = 0;
        for pair in self.paths() {
            let &[c1, c2] = pair else { unreachable!() };
            common_edges += other.has_path(c1, c2) as u16;
        }
        common_edges += other.has_path(self[0], *self.last().unwrap()) as u16;

        num_cities - common_edges
    }

    fn paths(&self) -> Windows<CityIndex> {
        self.windows(2)
    }

    fn update_pheromone(&self, delta_tau_matrix: &mut SquareMatrix<Float>, delta_tau: Float) {
        for path in self.paths() {
            let &[c1, c2] = path else { unreachable!() };
            delta_tau_matrix[reverse_order(c1.into(), c2.into())] += delta_tau;
        }
        // Last path.
        delta_tau_matrix[reverse_order(self[0].into(), self.last().unwrap().into())] += delta_tau;
    }

    fn get_hack_cvg(&self) -> Float {
        // Length is stored as two CityIndex'es in big endian order (i. e. u16 at [end]
        // is the lower end and u16 at [end - 1] is the higher end)
        let cvg_part_1: u16 = self[self.len() - 1].into();
        let cvg_part_2: u16 = self[self.len() - 2].into();
        let [b3, b4] = cvg_part_1.to_be_bytes();
        let [b1, b2] = cvg_part_2.to_be_bytes();
        Float::from_be_bytes([b1, b2, b3, b4])
    }
}

impl Index<TourIndex> for Tour {
    type Output = CityIndex;

    fn index(&self, index: TourIndex) -> &Self::Output {
        &self.cities[usize::from(index)]
    }
}

impl Index<TourIndex> for [CityIndex] {
    type Output = CityIndex;

    fn index(&self, index: TourIndex) -> &Self::Output {
        &self[usize::from(index)]
    }
}
