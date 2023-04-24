use crate::tour::CityIndex;

// Returns (low, high).
pub fn order<T: Ord>(a: T, b: T) -> (T, T) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

// Returns (high, low).
pub fn reverse_order<T: Ord>(a: T, b: T) -> (T, T) {
    if a > b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Creates a new `Vec<CityIndex>` containing all cities in order.
pub fn all_cities(count: usize) -> Vec<CityIndex> {
    (0..count).map(CityIndex::new).collect()
}

/// Fills up the buffer with cities in order.
/// Empties `buf` before proceeding.
/// It is neccessary to take `Vec` instead of slice,
/// since new elements need to be inserted.
pub fn all_cities_fill(buf: &mut Vec<CityIndex>, count: usize) {
    buf.clear();
    (0..count).map(|c| buf.push(CityIndex::new(c)));
}
