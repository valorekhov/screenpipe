use reqwest::Client;
use anyhow::{Result, anyhow};
use log::{debug, error, info};
use crate::stt::SttEngine;
use hound::{WavSpec, WavWriter};
use std::{collections::HashMap, future::Future, io::Cursor, pin::Pin};

pub struct RestPipeEngine {
    url: String,
    headers: HashMap<String, String>,
}

impl RestPipeEngine {
    pub fn new(url: String, headers: HashMap<String, String>) -> Self {
        Self { url, headers }
    }

    async fn transcribe_with_restpipe(
        wav_file: &[u8],
        device: &str,
        url: &str,
        headers: &HashMap<String, String>
    ) -> Result<String, anyhow::Error> {
        let client = Client::new();
        
        debug!("Sending request to RestPipe API");
        let mut request = client
            .post(url)
            .body(wav_file.to_vec())
            .header("Content-Type", "audio/wav");

        for (key, value) in headers {
            request = request.header(key, value);
        }

        let response = request.send().await?;

        match response.status() {
            reqwest::StatusCode::OK => {
                debug!("Received successful response from RestPipe API");
                let transcription = response.text().await?;
                
                if transcription.is_empty() {
                    info!(
                        "device: {}, transcription is empty",
                        device
                    );
                } else {
                    info!(
                        "device: {}, transcription successful. length: {} characters",
                        device,
                        transcription.len()
                    );
                }

                Ok(transcription)
            }
            status => {
                let error_message = format!("RestPipe API error: HTTP {}", status);
                error!("{}", error_message);
                Err(anyhow!(error_message))
            }
        }
    }
}

impl SttEngine for RestPipeEngine {
    fn transcribe<'a>(
        &'a self,
        audio_data: &'a [f32],
        sample_rate: u32,
        device_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            debug!("Starting RestPipe transcription");

            // Create a WAV file in memory
            let mut cursor = Cursor::new(Vec::new());
            {
                let spec = WavSpec {
                    channels: 1,
                    sample_rate,
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

            Self::transcribe_with_restpipe(&wav_data, device_name, &self.url, &self.headers).await
        })
    }
}

