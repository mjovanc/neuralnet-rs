use crate::utils::math::Vector;

fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh(x: f64) -> f64 {
    x.tanh()
}

fn softmax(vector: &Vector) -> Vector {
    let max = vector.elements.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = vector.elements.iter().map(|&x| (x - max).exp()).sum();
    Vector::new(vector.elements.iter().map(|&x| (x - max).exp() / exp_sum).collect())
}