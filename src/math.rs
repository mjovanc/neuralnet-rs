#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn derivative(x: f64) -> f64 {
    x * (1.0 - x)
}