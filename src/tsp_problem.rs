use std::path::Path;

use std::ops::Range;
use std::str::FromStr;
use tspf::WeightKind;
use tspf::{Tsp, TspBuilder};

use rand::Rng;
use rand::SeedableRng;

use crate::config::{DistanceT, Zeroable};
use crate::distance_matrix::DistanceMatrix;
use crate::index::CityIndex;
use crate::tsplib::TspLibProblems;

// For randomly generated problems
const MIN_CITY_COORD: f64 = 0.0;
const MAX_CITY_COORD: f64 = 1000.0;

#[derive(Clone)]
pub struct TspProblem {
    name: String,
    cities: Vec<Point>,
    distances: DistanceMatrix,
    // This will be 0 for custom problems.
    solution_length: DistanceT,
}

impl TspProblem {
    pub fn random<R: Rng + SeedableRng>(city_count: usize, rng: &mut R) -> TspProblem {
        let (cities, distances) = generate_cities(city_count, rng);

        TspProblem {
            name: format!("random{}", rng.gen::<u32>()),
            cities,
            distances,
            solution_length: DistanceT::ZERO,
        }
    }

    pub fn from_file(path: impl AsRef<Path>) -> TspProblem {
        let tsp = TspBuilder::parse_path(path).unwrap();
        let name = tsp.name().clone();
        // a280 has two identical cities (171 and 172), so it needs special treatment.
        // For simplicity, disallow it.
        assert_ne!(
            name, "a280",
            "a280 has two identical cities, this can cause problems"
        );
        let number_of_cities = tsp.dim();
        // CityIndex is a u16, so we cannot have more cities.
        assert!(
            number_of_cities <= usize::from(u16::MAX),
            "Number of cities cannot be greater than 16384, is {number_of_cities}"
        );

        let cities = {
            let coord_map = tsp.node_coords();
            let mut cities = Vec::with_capacity(number_of_cities);
            for idx in 1..=number_of_cities {
                let tspf_point_pos = coord_map.get(&idx).unwrap().pos();

                // We only care about 2D.
                let point = Point::new(tspf_point_pos[0], tspf_point_pos[1]);

                cities.push(point);
            }

            cities
        };
        assert_eq!(cities.len(), number_of_cities);

        let distances = Self::calculate_distances(number_of_cities, &tsp);

        let solution_length = TspLibProblems::from_str(name.as_str()).unwrap() as u32 as DistanceT;

        TspProblem {
            name,
            cities,
            distances,
            solution_length,
        }
    }

    pub fn number_of_cities(&self) -> usize {
        self.cities.len()
    }

    pub fn distances(&self) -> &DistanceMatrix {
        &self.distances
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn solution_length(&self) -> DistanceT {
        self.solution_length
    }

    fn calculate_distances(city_count: usize, tsp: &Tsp) -> DistanceMatrix {
        let mut distances = DistanceMatrix::new(city_count);

        // tspf indices start from 1
        for ind1 in 1..=city_count {
            for ind2 in (ind1 + 1)..=city_count {
                let dist_f64 = tsp.weight(ind1, ind2);
                // TSPLIB problem a280 has two identical cities (171 and 172), ignore this case.
                if tsp.name() != "a280" {
                    // tsp.weight() returns 0.0 on error.
                    assert!(
                        dist_f64 > 0.0,
                        "Error calculating distance, tsp.weight({ind1}, {ind2}) returned {dist_f64}"
                    );
                }
                // tsp.weight() returns unrounded distances. Distances are defined
                // to be u32 in TSPLIB95 format, and rounding depends on edge weight type.
                let dist = match tsp.weight_kind() {
                    WeightKind::Explicit => unimplemented!(),
                    WeightKind::Euc2d => nint(dist_f64),
                    WeightKind::Euc3d => unimplemented!(),
                    WeightKind::Max2d => unimplemented!(),
                    WeightKind::Max3d => unimplemented!(),
                    WeightKind::Man2d => unimplemented!(),
                    WeightKind::Man3d => unimplemented!(),
                    WeightKind::Ceil2d => unimplemented!(),
                    WeightKind::Geo => dist_f64 as u32,
                    WeightKind::Att => {
                        let d = nint(dist_f64);
                        if (f64::from(d)) < dist_f64 {
                            d + 1
                        } else {
                            d
                        }
                    }
                    WeightKind::Xray1 => unimplemented!(),
                    WeightKind::Xray2 => unimplemented!(),
                    WeightKind::Custom => unimplemented!(),
                    WeightKind::Undefined => unimplemented!(),
                };
                distances.set_dist(
                    CityIndex::new(ind1 as u16 - 1),
                    CityIndex::new(ind2 as u16 - 1),
                    dist as DistanceT,
                );
            }
        }
        distances
    }
}

// Same as nint() function defined in TSPLIB95 format.
fn nint(f: f64) -> u32 {
    (f + 0.5) as u32
}

// Returns city distance matrix.
fn generate_cities<R: Rng + SeedableRng>(
    city_count: usize,
    rng: &mut R,
) -> (Vec<Point>, DistanceMatrix) {
    let mut cities = Vec::with_capacity(city_count);

    // Generate some cities
    for _ in 0..city_count {
        let location = Point::random(rng, MIN_CITY_COORD..MAX_CITY_COORD);
        cities.push(location);
    }

    // Calculate distances (Euclidean plane)
    let mut distances = DistanceMatrix::new(city_count);

    for i in 0..city_count {
        for j in (i + 1)..city_count {
            let distance_f64 = Point::distance(cities[i], cities[j]);
            let distance = nint(distance_f64);
            distances.set_dist(
                CityIndex::new(i as u16),
                CityIndex::new(j as u16),
                distance as DistanceT,
            );
        }
    }

    (cities, distances)
}

#[derive(Debug, Clone, Copy)]
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Point {
        Point { x, y }
    }

    pub fn random<R: Rng + SeedableRng>(rng: &mut R, range: Range<f64>) -> Point {
        Point {
            x: rng.gen_range(range.clone()),
            y: rng.gen_range(range),
        }
    }

    pub fn distance(p1: Point, p2: Point) -> f64 {
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;

        (dx * dx + dy * dy).sqrt()
    }
}
