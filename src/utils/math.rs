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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_addition() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let v3 = v1 + v2;
        assert_eq!(v3.elements, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_dot_product() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(v1.dot(&v2), 32.0);
    }

    #[test]
    fn test_matrix_multiplication() {
        let m1 = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m2 = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let result = m1 * m2;
        assert_eq!(result.elements, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matrix_transpose() {
        let m1 = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let transposed = m1.transpose();
        assert_eq!(transposed.elements, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_matrix_inverse() {
        use approx::assert_relative_eq;

        let m = Matrix::new(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        if let Some(inverse) = m.inverse() {
            let identity = m * inverse;
            for (i, &x) in identity.elements.iter().enumerate() {
                assert_relative_eq!(x, if i % 3 == 0 { 1.0 } else { 0.0 }, epsilon = 1e-10);
            }
        } else {
            panic!("Matrix should have an inverse");
        }
    }

    #[test]
    fn test_matrix_determinant() {
        let m = Matrix::new(2, 2, vec![4.0, 7.0, 2.0, 6.0]);
        assert_eq!(m.determinant(), 10.0);
    }
}