use arrayfire::*;
use rand::{
    seq::{IteratorRandom, SliceRandom},
    thread_rng, Rng,
};

/*  Sizing Notation:
   k - size of current layer (input activations)
   j - size of next layer    (output activations)
   m - size of mini-batch
*/

#[derive(Clone)]
pub struct MNISTData {
    pub inputs: Array<f64>,          // always (784, 1) in size
    pub desired_outputs: Array<f64>, // always (10, 1) in size
}

// Holds error data for each layer
struct Nabla {
    nabla_w: Vec<Array<f64>>, // typically (j, k) in size
    nabla_b: Vec<Array<f64>>, // typically (j, 1) in size
}

pub struct Network {
    num_layers: usize,
    biases: Vec<Array<f64>>,  // typically (j, 1) in size
    weights: Vec<Array<f64>>, // typically (j, k) in size
}

impl Network {
    pub fn new(layer_dims: Vec<u64>) -> Self {
        let num_layers = layer_dims.len();

        let biases: Vec<Array<f64>> = layer_dims[1..num_layers]
            .iter()
            .map(|dim| randn::<f64>(Dim4::new(&[*dim, 1, 1, 1])))
            .collect::<Vec<Array<f64>>>();

        let weights: Vec<Array<f64>> = layer_dims[0..num_layers - 1]
            .iter()
            .enumerate()
            .map(|(index, _)| {
                randn::<f64>(Dim4::new(&[layer_dims[index + 1], layer_dims[index], 1, 1]))
            })
            .collect::<Vec<Array<f64>>>();

        Network {
            num_layers,
            biases,
            weights,
        }
    }

    pub fn SGD(
        &mut self,
        training_data: &MNISTData,
        epochs: usize,
        batches_per_epoch: usize,
        samples_per_batch: usize,
        learning_rate: f64,
        test_data: Option<&MNISTData>,
    ) {
        let mut rng = rand::thread_rng();
        (0..epochs).for_each(|j| {
            (0..batches_per_epoch).for_each(|_| {
                // println!("Total Input Dims: {}", training_data.inputs.dims());
                // println!(
                //     "Total Desired Output Dims: {}",
                //     training_data.desired_outputs.dims()
                // );

                // first randomly select samples from training_data
                let sample_indices: Vec<u64> = (0..samples_per_batch)
                    .map(|_| rng.gen_range(0..training_data.inputs.dims()[1]))
                    .collect();

                // generate input matrix
                let sample_indices_array: Array<u64> = Array::new(
                    &sample_indices,
                    Dim4::new(&[sample_indices.len() as u64, 1, 1, 1]),
                );
                let input_seq4gen: Seq<i32> = Seq::new(
                    0,
                    (training_data.inputs.dims()[0] - 1).try_into().unwrap(),
                    1,
                );
                let mut input_idxrs = Indexer::default();
                input_idxrs.set_index(&sample_indices_array, 1, None);
                input_idxrs.set_index(&input_seq4gen, 0, Some(false));
                let training_inputs: Array<f64> = index_gen(&training_data.inputs, input_idxrs);

                // generate output matrix
                let output_seq4gen: Seq<i32> = Seq::new(
                    0,
                    (training_data.desired_outputs.dims()[0] - 1)
                        .try_into()
                        .unwrap(),
                    1,
                );
                let mut input_idxrs = Indexer::default();
                input_idxrs.set_index(&sample_indices_array, 1, None);
                input_idxrs.set_index(&output_seq4gen, 0, Some(false));
                let training_outputs = index_gen(&training_data.desired_outputs, input_idxrs);

                // println!("Input Dims: {}", training_inputs.dims());
                // println!("Desired Output Dims: {}", training_outputs.dims());

                self.update_mini_batch(
                    training_inputs,
                    training_outputs,
                    learning_rate,
                    samples_per_batch,
                );
            });
            match test_data {
                Some(test) => println!(
                    "Epoch {}: {} / {}",
                    j,
                    self.evaluate(&test.inputs, &test.desired_outputs),
                    test.inputs.dims()[1]
                ),
                _ => println!("Epoch {} complete", j),
            }
        });
    }

    fn update_mini_batch(
        &mut self,
        training_inputs: Array<f64>,  // always (784, m) in size
        training_outputs: Array<f64>, // always (10, m) in size
        learning_rate: f64,
        samples_per_batch: usize,
    ) {
        // Execute backpropagation
        let nabla = self.backpropagate(training_inputs, training_outputs, samples_per_batch);

        // Update weights
        self.weights = self
            .weights
            .iter()
            .zip(nabla.nabla_w.iter())
            .map(|(w, nw)| {
                let step_size: f64 = learning_rate / samples_per_batch as f64;
                sub(w, &mul(&constant(step_size, nw.dims()), nw, false), false)
            })
            .collect();

        // Update biases
        self.biases = self
            .biases
            .iter()
            .zip(nabla.nabla_b.iter())
            .map(|(b, nb)| {
                let step_size: f64 = learning_rate / samples_per_batch as f64;
                sub(b, &mul(&constant(step_size, nb.dims()), nb, false), false)
            })
            .collect();
    }

    fn backpropagate(
        &mut self,
        training_inputs: Array<f64>,
        desired_outputs: Array<f64>,
        samples_per_batch: usize,
    ) -> Nabla {
        let mut nabla_w: Vec<Array<f64>> = Vec::with_capacity(self.num_layers);
        let mut nabla_b: Vec<Array<f64>> = Vec::with_capacity(self.num_layers);

        // feedforward
        let mut activations: Vec<Array<f64>> = Vec::new();
        activations.push(training_inputs);
        let mut zs: Vec<Array<f64>> = Vec::new();
        self.weights
            .iter()
            .zip(self.biases.iter())
            .enumerate()
            .for_each(|(layer, (w, b))| {
                let bias_matrix = tile(b, dim4!(1, samples_per_batch as u64, 1, 1)); // size (layer_dim, samples_per_batch)
                let z = add(
                    &matmul(w, &activations[layer], MatProp::NONE, MatProp::NONE),
                    &bias_matrix,
                    true,
                ); // size (layer+1_dim, samples_per_batch)
                zs.push(z);
                let activation = sigmoid(&zs[layer]); // size (layer+1_dim, samples_per_batch)
                activations.push(activation)
            });

        // backward pass (push in nabla values in reverse)
        let mut delta = mul(
            &self.cost_derivative(&activations[activations.len() - 1], &desired_outputs),
            &self.sigmoid_prime(&zs[zs.len() - 1]),
            false,
        ); // size (layer+1_dim, samples_per_batch)

        let sum_delta = sum(&delta, 1);
        nabla_b.push(sum_delta); // size (layeri+1_dim, samples_per_batch)

        nabla_w.push(matmul(
            &delta,
            &activations[activations.len() - 2],
            MatProp::NONE,
            MatProp::TRANS,
        )); // size (layeri+1_dim, layeri)

        for l in 2..self.num_layers {
            let z = &zs[zs.len() - l]; // first val zs[num_layers - 2] - second to last value
            let sp = self.sigmoid_prime(z);
            delta = mul(
                &matmul(
                    &self.weights[self.weights.len() - l + 1], // first val weights[num_layers - 1] - last value
                    &delta,
                    MatProp::TRANS,
                    MatProp::NONE,
                ),
                &sp,
                false,
            ); // size (layer+1_dim, samples_per_batch)

            let sum_delta = sum(&delta, 1);
            nabla_b.push(sum_delta);

            nabla_w.push(matmul(
                &delta,
                &activations[activations.len() - l - 1],
                MatProp::NONE,
                MatProp::TRANS,
            )); // size (layeri+1_dim, layeri);
        }

        // Reverse nabla_w and nabla_b
        let nabla_w = nabla_w.into_iter().rev().collect();
        let nabla_b = nabla_b.into_iter().rev().collect();

        Nabla { nabla_w, nabla_b }
    }

    pub fn evaluate(&mut self, test_inputs: &Array<f64>, desired_outputs: &Array<f64>) -> u32 {
        // Forward Prop
        let model_output = &self.feedforward(test_inputs);
        // println!("Model Output Dims: {}", model_output.dims());
        // Find max indices of model
        let (_model_max_vals, model_max_indices) = imax(model_output, 0);
        // print(&model_max_indices);
        // Find max index of desired output
        let (_desired_max_vals, desired_max_indices) = imax(desired_outputs, 0);
        // print(&desired_max_indices);
        // match indices
        let matching_array = eq(&desired_max_indices, &model_max_indices, false);
        // print(&matching_array);
        // find sum of match indices
        let sum_matching_array = sum(&matching_array, 1);
        // print(&sum_matching_array);
        // return matching sum
        let mut sum_host = vec![0; 1];
        sum_matching_array.host(&mut sum_host);
        sum_host[0]
    }

    fn feedforward(&self, a: &Array<f64>) -> Array<f64> {
        let mut activation_output = a.clone();
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            activation_output = sigmoid(&add(
                &matmul(w, &activation_output, MatProp::NONE, MatProp::NONE),
                b,
                true,
            ));
        }
        activation_output
    }

    fn cost_derivative(
        &self,
        output_activations: &Array<f64>,
        desired_output: &Array<f64>,
    ) -> Array<f64> {
        sub(output_activations, desired_output, false)
    }

    fn sigmoid_prime(&self, z: &Array<f64>) -> Array<f64> {
        let sigmoid_z = &sigmoid(z);
        let subtraction = &sub(&constant(1.0, sigmoid_z.dims()), sigmoid_z, false);
        let mutliplication = mul(sigmoid_z, subtraction, false);
        mutliplication
    }
}
