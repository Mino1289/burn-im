//! Basic image classification inference using a pretrained `ResNet-50` model.
//!
//! By default, this example looks for an image in `examples/data/` (e.g. `examples/data/bus.jpg`).
//! You can also pass a custom image path as a command-line argument.
//!
//! # Usage
//!
//! ```bash
//! # Run inference on the default image in examples/data/:
//! cargo run --release --example resnet50 --features image
//!
//! # Or specify a custom image path:
//! cargo run --release --example resnet50 --features image -- path/to/image.jpg
//! ```
//!
//! Pretrained weights are downloaded from the Hugging Face Hub on first run
//! and cached locally.

use std::path::{Path, PathBuf};

use burn_im::data::{ImageProcessor, imagenet_label};
use burn_im::prelude::*;
use burn_im::{CreateModelOptions, create_model};

/// Resolves the image path to use for inference.
///
/// Priority:
/// 1. Command-line argument (if provided).
/// 2. `examples/data/bus.jpg` (if present).
/// 3. Any supported image file found inside `examples/data/`.
fn resolve_image_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    // 1. If an image path was provided on the command line, use it.
    if let Some(arg) = std::env::args().nth(1) {
        let path = PathBuf::from(arg);
        if path.exists() {
            return Ok(path);
        }
        return Err(format!("Specified image file not found: {}", path.display()).into());
    }

    // 2. Otherwise, check for standard default image `examples/data/bus.jpg`.
    let default_bus = PathBuf::from("examples/data/bus.jpg");
    if default_bus.exists() {
        return Ok(default_bus);
    }

    // 3. Scan the `examples/data/` directory for any image file.
    let data_dir = Path::new("examples/data");
    if data_dir.is_dir() {
        let valid_extensions = ["jpg", "jpeg", "png", "webp", "bmp"];
        if let Ok(entries) = std::fs::read_dir(data_dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if let Some(ext) = p.extension().and_then(|e| e.to_str())
                    && valid_extensions.contains(&ext.to_lowercase().as_str())
                {
                    return Ok(p);
                }
            }
        }
    }

    Err(
        "No image found. Place an image in 'examples/data/' or pass the path as an argument:\n  \
         cargo run --example resnet50 --features image -- <path/to/image.jpg>"
            .into(),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image_path = resolve_image_path()?;
    println!("Selected image: {}", image_path.display());

    // 1. Select the computation device (CPU via Flex backend by default).
    let device = Device::flex();

    // 2. Load the pretrained ResNet-50 model from the Hugging Face Hub (cached locally).
    let model_name = "resnet50.a1_in1k";
    println!("Loading pretrained model '{model_name}'...");
    let model = create_model(model_name, &CreateModelOptions::pretrained(), &device)?;

    // 3. Configure the image preprocessor using the model's pretrained recipe
    // (target resolution, crop percentage, normalization mean & std).
    let cfg = model.pretrained_cfg().cloned().unwrap_or_default();
    let processor = ImageProcessor::from(&cfg);

    // 4. Load and preprocess the image (resize, centre crop, RGB tensor conversion, normalization).
    println!("Loading and preprocessing image...");
    let image = image::open(&image_path)?;
    let input = processor.preprocess(&image, &device);

    // 5. Run inference and compute softmax probabilities across ImageNet-1k classes.
    println!("Running inference...");
    let logits = model.forward(input);
    let probs = burn::tensor::activation::softmax(logits, 1);

    // 6. Extract the output tensor into a slice and sort predictions descending by probability.
    let data = probs.into_data();
    let values = data.as_slice::<f32>()?;
    let mut ranked: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    // 7. Print the top 5 predictions.
    println!("\nTop 5 predictions:");
    for (rank, (index, prob)) in ranked.into_iter().take(5).enumerate() {
        let label = imagenet_label(index).unwrap_or("<unknown>");
        println!(
            "  {}. {:>6.2}% - {} (class {})",
            rank + 1,
            prob * 100.0,
            label,
            index
        );
    }

    Ok(())
}
