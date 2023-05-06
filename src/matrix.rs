use std::{
    fmt::{self, Display},
    ops::{Index, IndexMut},
};

// Table layout:
// [row, row, ..]
#[derive(Debug, Clone)]
pub struct SquareMatrix<T>
where
    T: Copy,
{
    data: Vec<T>,
    side_length: usize,
}

impl<T> SquareMatrix<T>
where
    T: Copy,
{
    pub fn new(side_length: usize, init_value: T) -> SquareMatrix<T> {
        let data = vec![init_value; side_length * side_length];

        SquareMatrix { data, side_length }
    }

    pub fn side_length(&self) -> usize {
        self.side_length
    }

    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }

    pub fn row_mut(&mut self, y: usize) -> &mut [T] {
        &mut self.data[(self.side_length * y)..(self.side_length * (y + 1))]
    }

    pub fn row(&self, y: usize) -> &[T] {
        &self.data[(self.side_length * y)..(self.side_length * (y + 1))]
    }
}

impl<T> Index<(usize, usize)> for SquareMatrix<T>
where
    T: Copy,
{
    type Output = T;

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        &self.data[self.side_length * y + x]
    }
}

impl<T> IndexMut<(usize, usize)> for SquareMatrix<T>
where
    T: Copy,
{
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        &mut self.data[self.side_length * y + x]
    }
}

/// Trait used for matrix display. Although not stated
/// here, types implementing this should also support formatting
/// `{.<number>}`.
pub trait FloatDisplay: Copy + Display {}

impl FloatDisplay for f32 {}
impl FloatDisplay for f64 {}

impl<T: FloatDisplay> Display for SquareMatrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("--- SquareMatrix<f64> ---\n")?;
        for row in self.data.chunks_exact(self.side_length) {
            for elem in row {
                write!(f, "{elem:.2}\t")?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}
