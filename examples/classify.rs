//! Classify one image with a pretrained `timm` checkpoint.
//!
//! ```bash
//! cargo run --release --example classify --features image -- \
//!     path/to/image.jpg --model resnet50.a1_in1k --top 5
//! ```
//!
//! Weights are downloaded from the Hugging Face Hub on first run and cached locally.

use burn_im::data::{ImageProcessor, imagenet_label};
use burn_im::prelude::*;
use burn_im::{CreateModelOptions, create_model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = match args.next() {
        Some(path) if !path.starts_with("--") => path,
        _ => {
            eprintln!("usage: classify <image-path> [--model <name>] [--top <k>]");
            std::process::exit(2);
        }
    };

    let mut model_name = "resnet50.a1_in1k".to_string();
    let mut top = 5;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model_name = args.next().ok_or("--model needs a value")?,
            "--top" => top = args.next().ok_or("--top needs a value")?.parse()?,
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }

    let device = Device::flex();
    let model = create_model(&model_name, &CreateModelOptions::pretrained(), &device)?;
    let cfg = model.pretrained_cfg().cloned().unwrap_or_default();
    let processor = ImageProcessor::from(&cfg);

    let image = image::open(&path)?;
    let input = processor.preprocess(&image, &device);
    let probs = burn::tensor::activation::softmax(model.forward(input), 1);

    let data = probs.into_data();
    let values = data.as_slice::<f32>()?;
    let mut ranked: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    println!("{model_name}: {path}");
    for (index, prob) in ranked.into_iter().take(top) {
        let label = imagenet_label(index).unwrap_or("<unknown>");
        println!("{:>6.2}%  {label:<30} ({index})", prob * 100.0);
    }
    Ok(())
}
