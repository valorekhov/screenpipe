use reqwest::Client;
use serde_json::Value;


use anyhow::Result;
use log::{debug, error, info};
#[cfg(target_os = "macos")]
use objc::rc::autoreleasepool;

use crate::stt::{create_wav, SttEngine};

use std::{future::Future, pin::Pin};

pub struct DeepgramEngine {
    api_key: String,
}

// // Replace the get_deepgram_api_key function with this:
// fn get_deepgram_api_key() -> String {
//     "7ed2a159a094337b01fd8178b914b7ae0e77822d".to_string()
// }

impl DeepgramEngine {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }
    async fn transcribe_with_deepgram(
        api_key: &str,
        audio_data: &[f32],
        device: &str,
        sample_rate: u32,
        channels: u16,
    ) -> Result<String> {
        debug!("starting deepgram transcription");
        let client = Client::new();

        // Get the WAV data from the cursor
        let wav_data = create_wav(&audio_data, sample_rate, channels, cpal::SampleFormat::F32)?;

        let response = client
            .post("https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true")
            .header("Content-Type", "audio/wav")
            .header("Authorization", format!("Token {}", api_key))
            .body(wav_data)
            .send();

        match response.await {
            Ok(resp) => {
                debug!("received response from deepgram api");
                match resp.json::<Value>().await {
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
    fn transcribe<'a>(
        &'a self,
        audio_data: &'a [f32],
        sample_rate: u32,
        channels: u16,
        device_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
            Box::pin(async move {
                info!(
                    "device: {}, using deepgram api key: {}...",
                    device_name,
                    &self.api_key[..8]
                );
                DeepgramEngine::transcribe_with_deepgram(&self.api_key, audio_data, device_name, sample_rate, channels).await
        })
    }
}