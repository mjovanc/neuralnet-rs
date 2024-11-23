use crate::utils::math::Vector;

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn tanh(x: f64) -> f64 {
    if x > 10.0 {
        1.0
    } else if x < -10.0 {
        -1.0
    } else {
        let epx = x.exp();
        let emx = (-x).exp();
        (epx - emx) / (epx + emx)
    }
}

fn softmax(vector: &Vector) -> Vector {
    let max = vector.elements.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = vector.elements.iter().map(|&x| (x - max).exp()).sum();
    Vector::new(vector.elements.iter().map(|&x| (x - max).exp() / exp_sum).collect())
}

#[cfg(test)]
mod tests {
    use approx::{assert_relative_eq, assert_ulps_eq};
    use super::*;

    #[test]
    fn test_relu() {
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
        assert_eq!(relu(1.0), 1.0);
        assert_eq!(relu(2.0), 2.0);
    }

    #[test]
    fn test_sigmoid() {
        assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-10);
        assert_relative_eq!(sigmoid(-10.0), 4.5397868702434395e-5, epsilon = 1e-10);
        assert_relative_eq!(sigmoid(10.0), 0.9999546021312976, epsilon = 1e-10);
    }

    #[test]
    fn test_tanh() {
        assert_relative_eq!(tanh(-10.0), -0.9999999958776926, epsilon = 1e-10);
        assert_ulps_eq!(tanh(0.0), 0.0, max_ulps = 4);
        assert_relative_eq!(tanh(10.0), 0.9999999958776926, epsilon = 1e-10);
    }

    #[test]
    fn test_softmax() {
        use approx::assert_relative_eq;

        let input = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]);
        let result = softmax(&input);
        let sum: f64 = result.elements.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);  // Softmax should sum to 1, allowing for small floating-point errors
        assert!(result.elements[3] > result.elements[0]);  // Check if higher value produces higher output
    }
}