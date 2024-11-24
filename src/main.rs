use rand::seq::IndexedRandom;
use rand::thread_rng;
use crate::neural_network::NeuralNetwork;

mod neural_network;
mod math;
mod data;

fn main() {
    let d = data::get_data().unwrap();

    let inputs = d.training_inputs;
    let outputs = d.training_outputs;
    let test_inputs = d.test_inputs;

    // Print the first few data points to inspect
    println!("First 5 training inputs: {:?}", &inputs[..5]);
    println!("First 5 training outputs: {:?}", &outputs[..5]);

    // Initialize the network
    let mut neural_net = NeuralNetwork::new(0.3);

    // Train for 10000 epochs
    let (losses, accuracy) = neural_net.train(inputs, outputs, 10000);

    // Print the loss and accuracy
    let mut rng = thread_rng();
    let random_losses: Vec<_> = losses.choose_multiple(&mut rng, 3).collect();
    let random_accuracies: Vec<_> = accuracy.choose_multiple(&mut rng, 3).collect();

    println!("Random Losses: {:?}", random_losses);
    println!("Random Accuracies: {:?}", random_accuracies);

    for input in test_inputs.iter() {
        let prediction = neural_net.predict(input);
        println!("Input: {:?}, Prediction: {:.1}", input, prediction);
    }
}