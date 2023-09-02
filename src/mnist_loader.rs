use arrayfire::transpose;
use arrayfire::Array;
use arrayfire::Dim4;
use byteorder::BigEndian;
use byteorder::ReadBytesExt;
use flate2::read::GzDecoder;
use std::{
    fs::File,
    io::{Cursor, Read},
};

use crate::network::MNISTData;

#[derive(Debug)]
struct MNISTRawData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl MNISTRawData {
    fn new(f: &File) -> Result<MNISTRawData, std::io::Error> {
        let mut gz = GzDecoder::new(f);
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;
        let mut r = Cursor::new(&contents);

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        match magic_number {
            2049 => {
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
                sizes.push(r.read_i32::<BigEndian>()?);
            }
            _ => panic!(),
        }

        r.read_to_end(&mut data)?;

        Ok(MNISTRawData { sizes, data })
    }
}

pub fn load_data(dataset_name: &str) -> Result<MNISTData, std::io::Error> {
    let filename = format!("data\\{}-labels-idx1-ubyte.gz", dataset_name);
    let label_data = &MNISTRawData::new(&(File::open(filename))?)?;
    let filename = format!("data\\{}-images-idx3-ubyte.gz", dataset_name);
    let images_data = &MNISTRawData::new(&(File::open(filename))?)?;
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;

    let mut vector_inputs: Vec<f64> = Vec::new();
    for i in 0..images_data.sizes[0] as usize {
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();
        let mut image_data: Vec<f64> = image_data.into_iter().map(|x| x as f64 / 255.).collect();
        vector_inputs.append(&mut image_data);
    }
    let inputs: Array<f64> = Array::new(
        &vector_inputs,
        Dim4::new(&[image_shape as u64, images_data.sizes[0] as u64, 1, 1]),
    );
    let inputs = transpose(&inputs, false);

    let labels: Vec<u8> = label_data.data.clone();
    let vector_outputs: Vec<f64> = labels
        .iter()
        .flat_map(|label| {
            let mut label_vector: [f64; 10] = [0.0; 10];
            label_vector[*label as usize] = 1.0;
            label_vector
        })
        .collect();
    let desired_outputs: Array<f64> = Array::new(
        &vector_outputs,
        Dim4::new(&[10, images_data.sizes[0] as u64, 1, 1]),
    );
    let desired_outputs = transpose(&desired_outputs, false);

    println!(
        "{name} Inputs Dims: {inputs_dim}, {name} Outputs Dims: {outputs_dim}",
        name = dataset_name,
        inputs_dim = inputs.dims(),
        outputs_dim = desired_outputs.dims()
    );

    Ok(MNISTData {
        inputs,
        desired_outputs,
    })
}
