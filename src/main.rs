use crate::neural_network::NeuralNetwork;

mod neural_network;
mod math;
mod data;

fn main() {
    let d = data::get_data().unwrap();

    let inputs = d.training_inputs;
    let outputs = d.training_outputs;
    let test_inputs = d.test_inputs;

    // Initialize the network
    let mut neural_net = NeuralNetwork::new();

    // Train for 10000 epochs
    neural_net.train(inputs, outputs, 10000);

    for input in test_inputs.iter() {
        // Pass a set of inputs (two numbers) and get a prediction back which should be a sum of the two numbers
        let prediction = neural_net.predict(input);
        println!("Input: {:?}, Prediction: {:.1}", input, prediction);
    }
}