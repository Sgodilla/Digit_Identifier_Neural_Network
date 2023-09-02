mod mnist_loader;
mod network;
use network::Network;

fn main() {
    println!("Loading MNIST Data...");
    let training_data = mnist_loader::load_data("train").unwrap();
    let test_data = mnist_loader::load_data("t10k").unwrap();
    println!("Finished Loading MNIST Data!");

    let mut network = Network::new(vec![
        training_data.inputs.dims()[1],
        30,
        training_data.desired_outputs.dims()[1],
    ]);
    network.stochastic_gradient_descent(&training_data, 100, 100, 3.0, Some(&test_data));
    println!(
        "Final Test: {} / {}",
        network.evaluate(&test_data.inputs, &test_data.desired_outputs),
        test_data.desired_outputs.dims()[0]
    );
}
