use reqwest::blocking::Client;
use serde_json::Value;

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

pub struct DeepgramEngine {
    api_key: String,
}

// Replace the get_deepgram_api_key function with this:
fn get_deepgram_api_key() -> String {
    "7ed2a159a094337b01fd8178b914b7ae0e77822d".to_string()
}

impl DeepgramEngine {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }
    // TODO: this should use async reqwest not blocking, cause crash issue because all our code is async
    fn transcribe_with_deepgram(
        api_key: &str,
        audio_data: &[f32],
        device: &str,
        sample_rate: u32,
    ) -> Result<String> {
        debug!("starting deepgram transcription");
        let client = Client::new();

        // Create a WAV file in memory
        let mut cursor = Cursor::new(Vec::new());
        {
            let spec = WavSpec {
                channels: 1,
                sample_rate: sample_rate / 3, // for some reason 96khz device need 32 and 48khz need 16 (be mindful resampling)
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer = WavWriter::new(&mut cursor, spec)?;
            for &sample in audio_data {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        }

        // Get the WAV data from the cursor
        let wav_data = cursor.into_inner();

        let response = client
            .post("https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true")
            .header("Content-Type", "audio/wav")
            .header("Authorization", format!("Token {}", api_key))
            .body(wav_data)
            .send();

        match response {
            Ok(resp) => {
                debug!("received response from deepgram api");
                match resp.json::<Value>() {
                    Ok(result) => {
                        debug!("successfully parsed json response");
                        if let Some(err_code) = result.get("err_code") {
                            error!(
                                "deepgram api error code: {:?}, result: {:?}",
                                err_code, result
                            );
                            return Err(anyhow::anyhow!("Deepgram API error: {:?}", result));
                        }
                        let transcription = result["results"]["channels"][0]["alternatives"][0]
                            ["transcript"]
                            .as_str()
                            .unwrap_or("");

                        if transcription.is_empty() {
                            info!(
                                "device: {}, transcription is empty. full response: {:?}",
                                device, result
                            );
                        } else {
                            info!(
                                "device: {}, transcription successful. length: {} characters",
                                device,
                                transcription.len()
                            );
                        }

                        Ok(transcription.to_string())
                    }
                    Err(e) => {
                        error!("Failed to parse JSON response: {:?}", e);
                        Err(anyhow::anyhow!("Failed to parse JSON response: {:?}", e))
                    }
                }
            }
            Err(e) => {
                error!("Failed to send request to Deepgram API: {:?}", e);
                Err(anyhow::anyhow!(
                    "Failed to send request to Deepgram API: {:?}",
                    e
                ))
            }
        }
    }
}

impl SttEngine for DeepgramEngine {
    fn transcribe(&self, audio_data: &[f32], sample_rate: u32, device_name: &str) -> Result<String> {
        info!(
            "device: {}, using deepgram api key: {}...",
            device_name,
            &self.api_key[..8]
        );
        DeepgramEngine::transcribe_with_deepgram(&self.api_key, audio_data, device_name, sample_rate)
    }
}