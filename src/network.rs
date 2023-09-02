use arrayfire::*;
use rand::Rng;

/*  Sizing Notation:
   j - size of current layer (input activations)
   k - size of next layer    (output activations)
   m - size of mini-batch
*/

#[derive(Clone)]
pub struct MNISTData {
    pub inputs: Array<f64>,          // always (1, 784) in size
    pub desired_outputs: Array<f64>, // always (1, 10) in size
}
pub struct Network {
    num_layers: usize,
    weights: Vec<Array<f64>>, // typically (j, k) in size
}

impl Network {
    pub fn new(layer_dims: Vec<u64>) -> Self {
        let num_layers = layer_dims.len();

        let weights: Vec<Array<f64>> = layer_dims[0..num_layers - 1]
            .iter()
            .enumerate()
            .map(|(index, _)| {
                randn::<f64>(Dim4::new(&[
                    layer_dims[index] + 1,
                    layer_dims[index + 1],
                    1,
                    1,
                ]))
            })
            .collect::<Vec<Array<f64>>>();

        // print(weights.last().unwrap());

        Network {
            num_layers,
            weights,
        }
    }

    pub fn stochastic_gradient_descent(
        &mut self,
        training_data: &MNISTData,
        epochs: usize,
        batches_per_epoch: usize,
        learning_rate: f64,
        test_data: Option<&MNISTData>,
    ) {
        let mut rng = rand::thread_rng();
        let samples_per_batch: usize = training_data.inputs.dims()[0] as usize / batches_per_epoch;
        (0..epochs).for_each(|j| {
            let random_indices: Vec<u64> = (0..samples_per_batch + batches_per_epoch)
                .map(|_| rng.gen_range(0..training_data.inputs.dims()[0]))
                .collect();
            (0..batches_per_epoch).for_each(|k| {
                // first randomly select samples from training_data
                let sample_indices = &random_indices[k..k + samples_per_batch];

                // generate input matrix
                let sample_indices_array: Array<u64> = Array::new(
                    sample_indices,
                    Dim4::new(&[sample_indices.len() as u64, 1, 1, 1]),
                );
                let input_seq4gen: Seq<i32> = Seq::new(
                    0,
                    (training_data.inputs.dims()[1] - 1).try_into().unwrap(),
                    1,
                );
                let mut input_idxrs = Indexer::default();
                input_idxrs.set_index(&sample_indices_array, 0, None);
                input_idxrs.set_index(&input_seq4gen, 1, Some(false));
                let training_inputs: Array<f64> = index_gen(&training_data.inputs, input_idxrs);

                // generate output matrix
                let output_seq4gen: Seq<i32> = Seq::new(
                    0,
                    (training_data.desired_outputs.dims()[1] - 1)
                        .try_into()
                        .unwrap(),
                    1,
                );
                let mut input_idxrs = Indexer::default();
                input_idxrs.set_index(&sample_indices_array, 0, None);
                input_idxrs.set_index(&output_seq4gen, 1, Some(false));
                let training_outputs = index_gen(&training_data.desired_outputs, input_idxrs);

                // println!("Input Dims: {}", training_inputs.dims());
                // println!("Desired Output Dims: {}", training_outputs.dims());

                // println!(
                // "Training Inputs Dims: {}, Training Outputs Dims: {}",
                // training_inputs.dims(),
                // training_outputs.dims()
                // );
                self.update_mini_batch(training_inputs, training_outputs, learning_rate);
            });

            // print(self.weights.last().unwrap());
            match test_data {
                Some(test) => println!(
                    "Epoch {}: {} / {}",
                    j,
                    self.evaluate(&test.inputs, &test.desired_outputs),
                    test.inputs.dims()[0]
                ),
                _ => println!("Epoch {} complete", j),
            }
        });
    }

    fn update_mini_batch(
        &mut self,
        training_inputs: Array<f64>, // always (m, 784) in size
        desired_outputs: Array<f64>, // always (m, 10) in size
        learning_rate: f64,
    ) {
        // Execute forwardpropagation
        let activation_outputs = self.feedforward(&training_inputs);

        // Execute backpropagation
        self.backpropagate(&activation_outputs, &desired_outputs, learning_rate);
    }

    fn feedforward(&self, input: &Array<f64>) -> Vec<Array<f64>> {
        let mut activations: Vec<Array<f64>> = Vec::with_capacity(self.num_layers);

        activations.push(input.copy());

        for (layer_index, w) in self.weights.iter().enumerate() {
            // println!("Activation Dims: {}", activations[layer_index].dims());
            let activation_with_bias: Array<f64> = Self::add_bias(&activations[layer_index]);
            // println!("Activation with bias Dims: {}", activation_with_bias.dims());
            // println!("Weights Dims: {}", w.dims());

            let activation_output = sigmoid(&matmul(
                &activation_with_bias,
                w,
                MatProp::NONE,
                MatProp::NONE,
            ));
            activations.push(activation_output);
        }
        activations
    }

    fn add_bias(activation: &Array<f64>) -> Array<f64> {
        join(
            1,
            &constant::<f64>(1f64, dim4!(activation.dims()[0], 1, 1, 1)),
            activation,
        )
    }

    fn derivative(out: &Array<f64>) -> Array<f64> {
        out * (1 - out)
    }

    fn backpropagate(
        &mut self,
        activation_outputs: &[Array<f64>],
        desired_outputs: &Array<f64>,
        learning_rate: f64,
    ) {
        let mut output = activation_outputs.last().unwrap();
        let mut error = output - desired_outputs;

        let m = desired_outputs.dims()[0] as i32;

        for layer_index in (0..self.num_layers - 1).rev() {
            let activation_output_with_bias = Self::add_bias(&activation_outputs[layer_index]);
            let delta = transpose(&(Self::derivative(output) * error), false);

            let tg = learning_rate
                * matmul(
                    &delta,
                    &activation_output_with_bias,
                    MatProp::NONE,
                    MatProp::NONE,
                );
            let gradient = -(tg) / m;
            self.weights[layer_index] += transpose(&gradient, false);

            output = &activation_outputs[layer_index];

            let err = &matmul(
                &transpose(&delta, false),
                &transpose(&self.weights[layer_index], false),
                MatProp::NONE,
                MatProp::NONE,
            );

            error = index(err, &[seq!(), seq!(1, output.dims()[1] as i32, 1)]);
        }
    }

    pub fn evaluate(&mut self, test_inputs: &Array<f64>, desired_outputs: &Array<f64>) -> u64 {
        // Forward Prop
        let activation_outputs = &self.feedforward(test_inputs);
        let model_output = activation_outputs.last().unwrap();
        let (_model_max_vals, model_max_indices) = imax(model_output, 1);
        let (_desired_max_vals, desired_max_indices) = imax(desired_outputs, 1);
        let (matches, _) = count_all(&eq(&model_max_indices, &desired_max_indices, false));
        matches
    }
}
