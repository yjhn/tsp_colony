use mpi::traits::Equivalence;
use rand::Rng;
use std::fmt::Debug;
use std::fmt::Display;
use std::ops::Sub;

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
