
use std::{
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Error as E, Result};
use candle::{Device, IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use chrono::Utc;
use hf_hub::{api::sync::Api, Repo, RepoType};
use log::{debug, error, info};
#[cfg(target_os = "macos")]
use objc::rc::autoreleasepool;
use rand::{distributions::Distribution, SeedableRng};
use tokenizers::Tokenizer;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use candle_transformers::models::whisper::{self as m, audio, Config};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::{
    encode_single_audio, multilingual, stt::SttEngine, vad_engine::{SileroVad, VadEngine, VadEngineEnum, WebRtcVad}, AudioTranscriptionEngine
};

use hound::{WavSpec, WavWriter};
use std::io::Cursor;

use super::Model;

#[derive(Clone)]
pub struct WhisperModel {
    pub model: Model,
    pub tokenizer: Tokenizer,
    pub device: Device,
}

impl WhisperModel {
    pub fn new(engine: Arc<AudioTranscriptionEngine>) -> Result<Self> {
        debug!("Initializing WhisperModel");
        let device = Device::new_metal(0).unwrap_or(Device::new_cuda(0).unwrap_or(Device::Cpu));
        info!("device = {:?}", device);

        debug!("Fetching model files");
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let repo = match engine.as_ref() {
                AudioTranscriptionEngine::WhisperTiny => Repo::with_revision(
                    "openai/whisper-tiny".to_string(),
                    RepoType::Model,
                    "main".to_string(),
                ),
                AudioTranscriptionEngine::WhisperDistilLargeV3 => Repo::with_revision(
                    "distil-whisper/distil-large-v3".to_string(),
                    RepoType::Model,
                    "main".to_string(),
                ),
                _ => Repo::with_revision(
                    "openai/whisper-tiny".to_string(),
                    RepoType::Model,
                    "main".to_string(),
                ),
                // ... other engine options ...
            };
            let api_repo = api.repo(repo);
            let config = api_repo.get("config.json")?;
            let tokenizer = api_repo.get("tokenizer.json")?;
            let model = api_repo.get("model.safetensors")?;
            (config, tokenizer, model)
        };

        debug!("Parsing config and tokenizer");
        let config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        debug!("Loading model weights");
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &device)? };
        let model = Model::Normal(m::model::Whisper::load(&vb, config.clone())?);
        debug!("WhisperModel initialization complete");
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }
}