use std::ops::{Add, Mul};

// Vector structure
#[derive(Debug, Clone)]
pub struct Vector {
    pub(crate) elements: Vec<f64>,
}

// Matrix structure
#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    elements: Vec<f64>,
}

impl Vector {
    pub(crate) fn new(elements: Vec<f64>) -> Self {
        Vector { elements }
    }

    // Dot product
    fn dot(&self, other: &Vector) -> f64 {
        self.elements.iter().zip(other.elements.iter()).map(|(a, b)| a * b).sum()
    }
}

impl Matrix {
    fn new(rows: usize, cols: usize, elements: Vec<f64>) -> Self {
        assert_eq!(rows * cols, elements.len(), "Matrix dimensions do not match element count");
        Matrix { rows, cols, elements }
    }

    // Transpose
    fn transpose(&self) -> Self {
        let mut result = vec![0.0; self.elements.len()];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[j * self.rows + i] = self.elements[i * self.cols + j];
            }
        }
        Matrix::new(self.cols, self.rows, result)
    }

    // Naive inverse for 2x2 matrix (for simplicity)
    fn inverse(&self) -> Option<Self> {
        if self.rows != 2 || self.cols != 2 {
            return None; // Only for 2x2 matrices currently
        }
        let det = self.determinant();
        if det == 0.0 {
            return None; // No inverse if determinant is zero
        }
        let r = 1.0 / det;
        let new_data = vec![self.elements[3] * r, -self.elements[1] * r, -self.elements[2] * r, self.elements[0] * r];
        Some(Matrix::new(2, 2, new_data))
    }

    // Determinant for 2x2 matrix
    fn determinant(&self) -> f64 {
        if self.rows == 2 && self.cols == 2 {
            return self.elements[0] * self.elements[3] - self.elements[1] * self.elements[2];
        }
        panic!("Determinant not implemented for matrices other than 2x2");
    }
}

// Addition for vectors
impl Add for Vector {
    type Output = Vector;
    fn add(self, other: Vector) -> Vector {
        assert_eq!(self.elements.len(), other.elements.len(), "Vectors must have the same length");
        Vector::new(self.elements.iter().zip(other.elements.iter()).map(|(a, b)| a + b).collect())
    }
}

// Multiplication for matrices
impl Mul for Matrix {
    type Output = Matrix;
    fn mul(self, other: Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "Matrix dimensions incompatible for multiplication");
        let mut result = vec![0.0; self.rows * other.cols];
        for i in 0..self.rows {
            for j in 0..other.cols {
                result[i * other.cols + j] = (0..self.cols)
                    .map(|k| self.elements[i * self.cols + k] * other.elements[k * other.cols + j])
                    .sum();
            }
        }
        Matrix::new(self.rows, other.cols, result)
    }
}