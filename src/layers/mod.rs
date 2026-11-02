//! Shared neural building blocks reused across architecture families.

pub mod attention;
pub mod attention_pool;
pub mod conv_norm_act;
pub mod drop;
pub mod head;
pub mod mlp;
pub mod norm;
pub mod patch_embed;
pub mod pool;
pub mod pos_embed;
pub mod scale;

pub use attention::{Attention, AttentionMask};
pub use attention_pool::AttentionPoolLatent;
pub use conv_norm_act::{Activation, ConvNormAct, ConvNormActConfig};
pub use drop::{DropPath, PatchDropout, drop_path_schedule};
pub use head::{ClassifierHead, NormMlpClassifierHead};
pub use mlp::{Mlp, SwiGLU};
pub use norm::{GlobalResponseNorm, LayerNorm2d, RMSNorm2d, RmsNorm, RmsNorm2d};
pub use patch_embed::PatchEmbed;
pub use pool::{AttentionPoolPrr, PoolMode, SelectAdaptivePool2d};
pub use pos_embed::{
    GridIndexing, RotaryEmbeddingCat, RotaryEmbeddingDinoV3, RotaryEmbeddingMRope,
    RotaryEmbeddingMixed, apply_rot_embed_cat, resample_abs_pos_embed,
};
pub use scale::{LayerScale, LayerScale2d};
