use crate::tour::CityIndex;

// Returns (low, high).
pub fn order<T: Ord>(a: T, b: T) -> (T, T) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

pub fn all_cities(count: usize) -> Vec<CityIndex> {
    (0..count).map(CityIndex::new).collect()
}
