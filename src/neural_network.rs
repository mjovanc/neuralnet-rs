use rand::Rng;
use crate::math::{derivative, sigmoid};

pub struct NeuralNetwork {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let weights = vec![rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)];

        Self {
            weights,
            bias: rng.gen_range(0.0..1.0),
            learning_rate: 0.1,
        }
    }

    pub fn predict(&self, input: &[f64; 2]) -> f64 {
        let mut sum = self.bias;
        for (i, weight) in self.weights.iter().enumerate() {
            sum += input[i] * weight;
        }

        sigmoid(sum)
    }

    pub fn train(&mut self, inputs: Vec<[f64; 2]>, outputs: Vec<f64>, epochs: usize) {
        for _ in 0..epochs {
            for (i, input) in inputs.iter().enumerate() {

                // Get a prediction for a given input
                let output = self.predict(input);

                // Compute the difference between the actual and the desired output
                let error = outputs[i] - output;

                // Find the gradient of the loss function
                // (sort of like a hint about the direction to adjust the weights in)
                let delta = derivative(output);

                // Adjust the weights and the bias to reduce error in the output
                for j in 0..self.weights.len() {
                    self.weights[j] += self.learning_rate * error * input[j] * delta;
                }

                self.bias += self.learning_rate * error * delta;
            }
        }
    }
}
