use rand::Rng;
use crate::math::{derivative, sigmoid};
use plotters::prelude::*;

pub struct NeuralNetwork {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

impl NeuralNetwork {
    pub fn new(learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let weights = vec![rng.gen_range(0.0..1.0), rng.gen_range(0.0..1.0)];

        Self {
            weights,
            bias: rng.gen_range(0.0..1.0),
            learning_rate,
        }
    }

    pub fn predict(&self, input: &[f64; 2]) -> f64 {
        let mut sum = self.bias;
        for (i, weight) in self.weights.iter().enumerate() {
            sum += input[i] * weight;
        }

        sigmoid(sum)
    }

    pub fn train(&mut self, inputs: Vec<[f64; 2]>, outputs: Vec<f64>, epochs: usize) -> (Vec<f64>, Vec<f64>) {
        let mut losses = Vec::new();
        let mut accuracies = Vec::new();

        println!("Learning rate: {}", self.learning_rate);

        for _ in 0..epochs {
            let mut total_loss = 0.0;
            let mut correct_predictions = 0;

            for (i, input) in inputs.iter().enumerate() {

                // Get a prediction for a given input
                let output = self.predict(input);

                // Compute the difference between the actual and the desired output
                let error = outputs[i] - output;

                // Mean Squared Error
                total_loss += error.powi(2);

                // Find the gradient of the loss function
                // (sort of like a hint about the direction to adjust the weights in)
                let delta = derivative(output);

                // Adjust the weights and the bias to reduce error in the output
                for j in 0..self.weights.len() {
                    self.weights[j] += self.learning_rate * error * input[j] * delta;
                }

                self.bias += self.learning_rate * error * delta;

                // Calculate accuracy
                let predicted = if output >= 0.5 { 1.0 } else { 0.0 };
                if predicted == outputs[i] {
                    correct_predictions += 1;
                }
            }

            let loss = total_loss / inputs.len() as f64;
            let accuracy = correct_predictions as f64 / inputs.len() as f64;

            losses.push(loss);
            accuracies.push(accuracy);
        }

        (losses, accuracies)
    }

    pub fn plot_loss_curve(&self, losses: &[f64]) {
        let root = BitMapBackend::new("loss_curve.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("Loss Curve", ("sans-serif", 50).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0..losses.len(), 0.0..*losses.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        chart.draw_series(LineSeries::new(
            (0..).zip(losses.iter()).map(|(x, y)| (x, *y)),
            &RED,
        )).unwrap();
    }

    pub fn plot_accuracy_curve(&self, accuracy: &[f64]) {
        let root = BitMapBackend::new("accuracy_curve.png", (640, 480)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let mut chart = ChartBuilder::on(&root)
            .caption("Accuracy Curve", ("sans-serif", 50).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0..accuracy.len(), 0.0..*accuracy.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        chart.draw_series(LineSeries::new(
            (0..).zip(accuracy.iter()).map(|(x, y)| (x, *y)),
            &BLUE,
        )).unwrap();
    }
}
