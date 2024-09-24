use reqwest::Client;
use anyhow::{Result, anyhow};
use log::{debug, error, info};
use crate::stt::{create_wav, resample, SttEngine};
use std::{collections::HashMap, future::Future, pin::Pin};

pub struct RestPipeEngine {
    url: String,
    headers: HashMap<String, String>,
    payload_field: Option<String>,
    resample_to_rate: Option<u32>,
}

impl RestPipeEngine {
    pub fn new(url: String, headers: HashMap<String, String>, payload_field: Option<String>, resample_to_rate: Option<u32>) -> Self {
        Self { url, headers, payload_field, resample_to_rate}
    }

    async fn transcribe_with_restpipe(
        wav_file: &[u8],
        device: &str,
        url: &str,
        headers: &HashMap<String, String>,
        payload_field: &Option<String>
    ) -> Result<String, anyhow::Error> {
        let client = Client::new();
        
        debug!("Sending request to RestPipe API {}, with wav_file lenght: {}", url, wav_file.len());
        let mut request = client.post(url);

        if let Some(payload_field) = payload_field {
            request = request.multipart(reqwest::multipart::Form::new()
                .part(payload_field.to_owned(),
                 reqwest::multipart::Part::bytes(wav_file.to_vec()).file_name("file.wav").mime_str("audio/wav")?));
        } else {
            request = request.body(wav_file.to_vec()).header("Content-Type", "audio/wav");
        }
        for (key, value) in headers {
            debug!("Request header: {} = {}", key, value);
            request = request.header(key, value);
        }

        let response = request.send().await;

        match response {
            Err(e) => {
                error!("Error sending request to RestPipe API: {}", e);
                return Err(anyhow!("Error sending request to RestPipe API: {}", e));
            }
            Ok(response) => {
                info!("Response status: {}", response.status());

                match response.status() {
                    reqwest::StatusCode::OK => {
                        debug!("Received successful response from RestPipe API");

                        let content_type = response.headers().get(reqwest::header::CONTENT_TYPE);
                        let transcription = if let Some(content_type) = content_type {
                            if content_type.to_str().unwrap_or("").starts_with("application/json") {
                                debug!("Response is JSON, parsing accordingly");
                                let json: serde_json::Value = response.json().await?;
                                // Assuming the transcription is in a "text" field, adjust as needed
                                let transcription = json["text"].as_str().unwrap_or("").to_string();
                                transcription
                            } else {
                                debug!("Response is not JSON, treating as plain text");
                                response.text().await?
                            }
                        } else {
                            debug!("No Content-Type header, treating response as plain text");
                            response.text().await?
                        };

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
    }
}

impl SttEngine for RestPipeEngine {
    fn transcribe<'a>(
        &'a self,
        audio_data: &'a [f32],
        sample_rate: u32,
        channels: u16,
        device_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        let mut new_channels = channels;
        Box::pin(async move {
            debug!("Starting RestPipe transcription for device: {}, incoming sample rate: {}", device_name, sample_rate);
            let data = if self.resample_to_rate.is_some(){
                if self.resample_to_rate.unwrap() != sample_rate {
                    debug!("Resampling audio data from {} to {} Hz", sample_rate, self.resample_to_rate.unwrap());
                    new_channels = 1; // the resumpled audio is mono
                    resample(audio_data.to_vec(), channels, sample_rate, self.resample_to_rate.unwrap())?
                } else {
                    audio_data.to_vec()
                }
            } else {
                audio_data.to_vec()
            };
            let wav_data = create_wav(&data, sample_rate, new_channels, cpal::SampleFormat::I16)?;
            Self::transcribe_with_restpipe(&wav_data, device_name, &self.url, 
                &self.headers, &self.payload_field).await
        })
    }}

