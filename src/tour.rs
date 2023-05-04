use crate::config::Float;
use crate::distance_matrix::DistanceMatrix;
use crate::matrix::SquareMatrix;
use crate::static_assert;
use crate::utils::all_cities;
use crate::utils::order;
use crate::utils::reverse_order;
use mpi::traits::Equivalence;
use rand::prelude::SliceRandom;
use rand::Rng;
use std::fmt::write;
use std::fmt::Debug;
use std::fmt::Display;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::mem::size_of;
use std::path::Path;
use std::slice::Windows;

/// Index of the city in the city matrix.
/// `u16` is enough since it is extremely unlikely that number of cities would be greater.
// TODO: it would be enough to use u16 here instead of usize
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

// TODO: where is this used?
impl From<&CityIndex> for usize {
    fn from(value: &CityIndex) -> Self {
        value.0.into()
    }
}

impl Debug for CityIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for CityIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "CityIndex({})", self.0)
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone)]
pub struct Tour {
    cities: Vec<CityIndex>,
    tour_length: u32,
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
        tour_length: u32::MAX,
    };

    pub fn number_of_cities(&self) -> usize {
        self.cities.len()
    }

    pub fn length(&self) -> u32 {
        self.tour_length
    }

    pub fn cities(&self) -> &[CityIndex] {
        &self.cities
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
    pub fn clone_from_cities(tour: &[CityIndex], length: u32, distances: &DistanceMatrix) -> Tour {
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
}

pub trait TourFunctions {
    fn calculate_tour_length(&self, distances: &DistanceMatrix) -> u32;

    fn get_hack_tour_length(&self) -> u32;

    fn get_hack_cvg(&self) -> Float;

    fn paths(&self) -> Windows<CityIndex>;

    fn has_path(&self, x: CityIndex, y: CityIndex) -> bool;

    fn distance(&self, other: &[CityIndex]) -> u16;

    // fn get_hack_mpi_rank(&self) -> i32;

    fn update_pheromone(&self, delta_tau_matrix: &mut SquareMatrix<Float>, delta_tau: Float);
}

impl TourFunctions for [CityIndex] {
    fn calculate_tour_length(&self, distances: &DistanceMatrix) -> u32 {
        assert!(self.len() > 1);

        let mut tour_length = 0;
        for pair in self.paths() {
            tour_length += distances[(pair[0], pair[1])];
        }
        // Add distance from last to first.
        tour_length += distances[(*self.last().unwrap(), self[0])];

        tour_length
    }

    // The length must first be inserted using Tour::hack_append_length_and_mpi_rank().
    fn get_hack_tour_length(&self) -> u32 {
        // Length is stored as two CityIndex'es in big endian order (i. e. u16 at [end]
        // is the lower end and u16 at [end - 1] is the higher end)
        let tour_length_part_1: u16 = self[self.len() - 3].into();
        let tour_length_part_2: u16 = self[self.len() - 4].into();
        // let tour_length_part_1: u16 = self[self.len() - 1].into();
        // let tour_length_part_2: u16 = self[self.len() - 2].into();
        let [b3, b4] = tour_length_part_1.to_be_bytes();
        let [b1, b2] = tour_length_part_2.to_be_bytes();
        u32::from_be_bytes([b1, b2, b3, b4])
    }

    fn has_path(&self, x: CityIndex, y: CityIndex) -> bool {
        let inner_paths = self.paths().any(|pair| {
            let &[c1, c2] = pair else { unreachable!() };
            order(x, y) == order(c1, c2)
        });
        inner_paths || order(x, y) == order(self[0], *self.last().unwrap())
    }

    /// Returns the number of paths (edges) in one tour that are not in other.
    /// Assumes that both tours have the same number of cities.
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

    // fn get_hack_mpi_rank(&self) -> i32 {
    //     // Length is stored as two CityIndex'es in big endian order (i. e. u16 at [end]
    //     // is the lower end and u16 at [end - 1] is the higher end)
    //     let rank_part_1: u16 = self[self.len() - 1].into();
    //     let rank_part_2: u16 = self[self.len() - 2].into();
    //     let [b3, b4] = rank_part_1.to_be_bytes();
    //     let [b1, b2] = rank_part_2.to_be_bytes();
    //     i32::from_be_bytes([b1, b2, b3, b4])
    // }

    fn update_pheromone(&self, delta_tau_matrix: &mut SquareMatrix<Float>, delta_tau: Float) {
        for path in self.windows(2) {
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
