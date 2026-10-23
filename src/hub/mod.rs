//! Hugging Face Hub access: resolution of `config.json` and `.safetensors` weights for
//! checkpoints published under the `timm/` organization.

pub mod config;
pub mod head_filter;
pub mod loader;

use std::path::PathBuf;

use config::TimmHfConfig;

/// Weights file published by `timm` on the Hub.
pub const WEIGHTS_FILE: &str = "model.safetensors";
/// Metadata file published by `timm` on the Hub.
pub const CONFIG_FILE: &str = "config.json";

/// Synchronous client for the Hugging Face Hub.
///
/// Downloads go through `hf-hub`'s `ureq` backend and are cached in the standard
/// `~/.cache/huggingface/hub` directory, so no async runtime is required.
pub struct Hub {
    api: hf_hub::api::sync::Api,
}

impl Hub {
    /// Creates a Hub client.
    pub fn new() -> crate::Result<Self> {
        let api =
            hf_hub::api::sync::Api::new().map_err(|err| crate::Error::HubError(err.to_string()))?;
        Ok(Self { api })
    }

    /// Repository id for an `<arch>.<tag>` pair, e.g. `timm/resnet50.a1_in1k`.
    pub fn repo_id(arch: &str, tag: &str) -> String {
        format!("timm/{arch}.{tag}")
    }

    /// Downloads (or resolves from cache) a file of a repository.
    pub fn download(&self, repo: &str, file: &str) -> crate::Result<PathBuf> {
        self.api
            .model(repo.to_string())
            .get(file)
            .map_err(|err| crate::Error::HubError(format!("{repo}/{file}: {err}")))
    }

    /// Downloads and parses `config.json`.
    pub fn config(&self, repo: &str) -> crate::Result<TimmHfConfig> {
        let path = self.download(repo, CONFIG_FILE)?;
        TimmHfConfig::from_file(path)
    }

    /// Downloads `model.safetensors`.
    pub fn weights(&self, repo: &str) -> crate::Result<PathBuf> {
        self.download(repo, WEIGHTS_FILE)
    }
}
