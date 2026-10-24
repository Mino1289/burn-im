//! Resize + centre-crop + normalize preprocessing, matching `timm`/PIL geometry.
//!
//! The geometry is the one `timm` uses at inference: the short side of the image is resized to
//! `floor(max(height, width) / crop_pct)` and the centre is then cropped to the model input size.
//! Resampling uses the `image` crate's bilinear (`Triangle`) or bicubic (`CatmullRom`) kernels:
//! geometrically equivalent to PIL, but not bit-identical (kernel support and rounding differ).

use crate::hub::config::{Interpolation, PretrainedCfg};

/// Pixel geometry of a resize-then-centre-crop, in source pixels.
///
/// Available without the `image` feature so the maths can be tested with no decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageGeometry {
    /// Size after the aspect-preserving resize, `(width, height)`.
    pub resize: (u32, u32),
    /// Top-left corner of the crop box, `(left, top)`.
    pub offset: (u32, u32),
    /// Crop size, equal to the model input `(width, height)`.
    pub crop: (u32, u32),
}

impl ImageGeometry {
    /// Computes the `timm`/PIL geometry for a `src_width × src_height` image.
    ///
    /// `crop_pct` is expected in `(0, 1]`, as published in [`PretrainedCfg`].
    pub fn new(src_width: u32, src_height: u32, width: u32, height: u32, crop_pct: f64) -> Self {
        // timm computes the scale side with `floor(size / crop_pct)`; torchvision resizes the
        // shortest edge to it and truncates the longest edge (`int(short * long_src / short_src)`).
        let short_side = (height.max(width) as f64 / crop_pct).floor() as u32;
        let resize = if src_width <= src_height {
            (
                short_side,
                (src_height as f64 * short_side as f64 / src_width as f64) as u32,
            )
        } else {
            (
                (src_width as f64 * short_side as f64 / src_height as f64) as u32,
                short_side,
            )
        };
        // torchvision `CenterCrop` rounds with Python's `round`, i.e. ties to even.
        let offset = (
            ((resize.0 as f64 - width as f64) / 2.0).round_ties_even() as u32,
            ((resize.1 as f64 - height as f64) / 2.0).round_ties_even() as u32,
        );
        Self {
            resize,
            offset,
            crop: (width, height),
        }
    }
}

/// Preprocessing pipeline (resize, centre crop, normalize) built from a [`PretrainedCfg`].
///
/// Missing config fields already fall back to the [`PretrainedCfg`] defaults (224×224,
/// `crop_pct = 0.875`, bicubic, ImageNet mean/std).
#[derive(Debug, Clone)]
// Fields are only read by the image-gated `preprocess`.
#[cfg_attr(not(feature = "image"), allow(dead_code))]
pub struct ImageProcessor {
    width: usize,
    height: usize,
    crop_pct: f64,
    interpolation: Interpolation,
    mean: [f64; 3],
    std: [f64; 3],
}

impl From<&PretrainedCfg> for ImageProcessor {
    fn from(cfg: &PretrainedCfg) -> Self {
        let (height, width) = cfg.input_hw();
        Self {
            width,
            height,
            crop_pct: cfg.crop_pct,
            interpolation: cfg.interpolation,
            mean: cfg.mean,
            std: cfg.std,
        }
    }
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self::from(&PretrainedCfg::default())
    }
}

impl ImageProcessor {
    /// Decodes nothing itself: resizes `image` to the recipe geometry, crops it and returns the
    /// normalized `[1, 3, H, W]` f32 tensor expected by [`crate::VisionModel::forward`].
    ///
    /// Resampling is not bit-identical to PIL; see the module docs.
    #[cfg(feature = "image")]
    pub fn preprocess(
        &self,
        image: &image::DynamicImage,
        device: &burn::tensor::Device,
    ) -> burn::tensor::Tensor<4> {
        use burn::tensor::{Tensor, TensorData};
        use image::imageops::FilterType;

        let rgb = image.to_rgb8();
        let geometry = ImageGeometry::new(
            rgb.width(),
            rgb.height(),
            self.width as u32,
            self.height as u32,
            self.crop_pct,
        );
        let filter = match self.interpolation {
            Interpolation::Nearest => FilterType::Nearest,
            Interpolation::Bilinear => FilterType::Triangle,
            Interpolation::Bicubic | Interpolation::Other => FilterType::CatmullRom,
        };
        let resized = image::imageops::resize(&rgb, geometry.resize.0, geometry.resize.1, filter);
        let cropped = image::imageops::crop_imm(
            &resized,
            geometry.offset.0,
            geometry.offset.1,
            geometry.crop.0,
            geometry.crop.1,
        )
        .to_image();

        let plane = self.width * self.height;
        let mut data = vec![0.0_f32; 3 * plane];
        for (x, y, pixel) in cropped.enumerate_pixels() {
            let index = y as usize * self.width + x as usize;
            for channel in 0..3 {
                let normalized =
                    (pixel[channel] as f64 / 255.0 - self.mean[channel]) / self.std[channel];
                data[channel * plane + index] = normalized as f32;
            }
        }
        Tensor::from_data(
            TensorData::new(data, [1, 3, self.height, self.width]),
            device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(width: u32, height: u32, crop_pct: f64) -> ImageGeometry {
        ImageGeometry::new(width, height, 224, 224, crop_pct)
    }

    #[test]
    fn default_landscape() {
        assert_eq!(
            geometry(500, 375, 0.875),
            ImageGeometry {
                resize: (341, 256),
                offset: (58, 16),
                crop: (224, 224),
            }
        );
    }

    #[test]
    fn default_square() {
        assert_eq!(
            geometry(224, 224, 0.875),
            ImageGeometry {
                resize: (256, 256),
                offset: (16, 16),
                crop: (224, 224),
            }
        );
    }

    #[test]
    fn default_wide() {
        assert_eq!(
            geometry(1000, 800, 0.875),
            ImageGeometry {
                resize: (320, 256),
                offset: (48, 16),
                crop: (224, 224),
            }
        );
    }

    #[test]
    fn default_portrait() {
        assert_eq!(
            geometry(375, 500, 0.875),
            ImageGeometry {
                resize: (256, 341),
                offset: (16, 58),
                crop: (224, 224),
            }
        );
    }

    #[test]
    fn custom_crop_pct_resizes_to_input_short_side() {
        assert_eq!(
            geometry(500, 375, 1.0),
            ImageGeometry {
                resize: (298, 224),
                offset: (37, 0),
                crop: (224, 224),
            }
        );
    }

    #[test]
    fn scale_side_floors_like_timm() {
        // timm: floor(224 / 0.9) == 248; rounding would give 249.
        assert_eq!(
            geometry(448, 448, 0.9),
            ImageGeometry {
                resize: (248, 248),
                offset: (12, 12),
                crop: (224, 224),
            }
        );
    }

    #[test]
    fn long_side_truncates_like_torchvision() {
        // torchvision: int(256 * 640 / 427) == 383; rounding would give 384.
        assert_eq!(
            geometry(640, 427, 0.875),
            ImageGeometry {
                resize: (383, 256),
                offset: (80, 16),
                crop: (224, 224),
            }
        );
    }

    #[test]
    fn non_square_crop_keeps_width_height_order() {
        assert_eq!(
            ImageGeometry::new(10, 8, 6, 4, 1.0),
            ImageGeometry {
                resize: (7, 6),
                offset: (0, 1),
                crop: (6, 4),
            }
        );
    }
}

#[cfg(all(test, feature = "image"))]
mod image_tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn preprocess_shape_layout_and_normalization() {
        let cfg = PretrainedCfg {
            input_size: [3, 4, 6],
            crop_pct: 1.0,
            interpolation: Interpolation::Bilinear,
            mean: [0.5; 3],
            std: [0.5; 3],
            ..PretrainedCfg::default()
        };
        let processor = ImageProcessor::from(&cfg);

        // Constant per-channel colours survive any resampling kernel unchanged.
        let image = image::RgbImage::from_pixel(10, 8, image::Rgb([64, 128, 192]));
        let input = processor.preprocess(&image::DynamicImage::ImageRgb8(image), &Device::flex());
        assert_eq!(input.dims(), [1, 3, 4, 6]);

        let data = input.into_data();
        let values = data.as_slice::<f32>().unwrap();
        let plane = 4 * 6;
        for (channel, value) in [64.0_f64, 128.0, 192.0].into_iter().enumerate() {
            let expected = ((value / 255.0 - 0.5) / 0.5) as f32;
            for index in 0..plane {
                assert!(
                    (values[channel * plane + index] - expected).abs() < 1e-6,
                    "channel {channel} index {index}: {} != {expected}",
                    values[channel * plane + index]
                );
            }
        }
    }
}
