pub mod whisper;
mod deepgram;
mod restpipe;

pub use deepgram::DeepgramEngine;
use restpipe::RestPipeEngine;
use whisper::{CandleWhisperModel, WhisperEngine};

use std::{
    collections::HashMap, path::PathBuf, sync::{atomic::{AtomicBool, Ordering}, Arc}, time::{SystemTime, UNIX_EPOCH}
};

use anyhow::Result;
use log::{debug, error, info};
#[cfg(target_os = "macos")]
use objc::rc::autoreleasepool;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use crate::{
    stt::{perform_stt, SttEngine}, vad_engine::{SileroVad, VadEngine, VadEngineEnum, WebRtcVad}, AudioInput, AudioTranscriptionEngine, TranscriptionResult, WhisperModel,
};

pub fn initialize_engines(
    local_model: Option<CandleWhisperModel>,
    api_url: Option<String>,
    api_headers: Option<String>,
    deepgram_api_key: Option<String>,
) -> Result<(Box<dyn SttEngine + Send + Sync>, Option<Box<dyn SttEngine + Send + Sync>>)> {
    let local_model = local_model.unwrap_or(CandleWhisperModel::Tiny);
    let primary_engine: Box<dyn SttEngine + Send + Sync> = if let Some(ref api_key) = deepgram_api_key {
        Box::new(DeepgramEngine::new(api_key.clone()))
    } else if let Some(ref url) = api_url {
        let api_headers: HashMap<String, String> = {
            let mut map = HashMap::new();
            if let Some(headers_arg) = api_headers {    
                for header in headers_arg.split(';') {
                    let parts: Vec<&str> = header.split(':').map(str::trim).collect();
                    if parts.len() == 2 {
                        map.insert(parts[0].to_string(), parts[1].to_string());
                    }
                }
            }
            map    
        };
        Box::new(RestPipeEngine::new(url.clone(), api_headers))
    } else {
        let whisper_model = match local_model {
            CandleWhisperModel::Tiny => AudioTranscriptionEngine::WhisperTiny,
            _ => AudioTranscriptionEngine::WhisperDistilLargeV3,
        };
        Box::new(WhisperEngine::new(WhisperModel::new(Arc::new(whisper_model))?).expect("Could not create the WhisperEngine"))
    };

    let fallback_engine: Option<Box<dyn SttEngine + Send + Sync>> = if deepgram_api_key.is_some() || api_url.is_some() {
        let whisper_model = match local_model {
            CandleWhisperModel::Tiny => AudioTranscriptionEngine::WhisperTiny,
            _ => AudioTranscriptionEngine::WhisperDistilLargeV3,
        };
        Some(Box::new(WhisperEngine::new(WhisperModel::new(Arc::new(whisper_model))?).expect("Could not create the WhisperEngine")))
    } else {
        None
    };

    Ok((primary_engine, fallback_engine))
}

pub async fn create_comm_channel(
    primary_whisper_engine: Box<dyn SttEngine + Send + Sync>,
    fallback_whisper_engine: Option<Box<dyn SttEngine + Send + Sync>>,
    vad_engine: VadEngineEnum,
    output_path: &Option<PathBuf>,
) -> Result<(
    UnboundedSender<AudioInput>,
    UnboundedReceiver<TranscriptionResult>,
    Arc<AtomicBool>, // Shutdown flag
)> {
    let (input_sender, mut input_receiver): (
        UnboundedSender<AudioInput>,
        UnboundedReceiver<AudioInput>,
    ) = unbounded_channel();
    let (output_sender, output_receiver): (
        UnboundedSender<TranscriptionResult>,
        UnboundedReceiver<TranscriptionResult>,
    ) = unbounded_channel();
    let mut vad_engine: Box<dyn VadEngine + Send> = match vad_engine {
        VadEngineEnum::WebRtc => Box::new(WebRtcVad::new()),
        VadEngineEnum::Silero => Box::new(SileroVad::new()?),
    };

    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let shutdown_flag_clone = shutdown_flag.clone();
    let output_path = output_path.clone();

    tokio::spawn(async move {
        loop {
            if shutdown_flag_clone.load(Ordering::Relaxed) {
                info!("Whisper channel shutting down");
                break;
            }
            debug!("Waiting for input from input_receiver");

            tokio::select! {
                Some(input) = input_receiver.recv() => {
                    debug!("Received input from input_receiver");
                    let timestamp = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("Time went backwards")
                        .as_secs();

                    let transcription_result = if cfg!(target_os = "macos") {
                        #[cfg(target_os = "macos")]
                        {
                            autoreleasepool(|| {
                                match perform_stt(&input, &*primary_whisper_engine, fallback_whisper_engine.as_deref(), &mut *vad_engine, &output_path).await {
                                    Ok((transcription, path)) => TranscriptionResult {
                                        input: input.clone(),
                                        transcription: Some(transcription),
                                        path,
                                        timestamp,
                                        error: None,
                                    },
                                    Err(e) => {
                                        error!("STT error for input {}: {:?}", input.device, e);
                                        TranscriptionResult {
                                            input: input.clone(),
                                            transcription: None,
                                            path: "".to_string(),
                                            timestamp,
                                            error: Some(e.to_string()),
                                        }
                                    },
                                }
                            })
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            unreachable!("This code should not be reached on non-macOS platforms")
                        }
                    } else {
                        match perform_stt(&input, &*primary_whisper_engine, fallback_whisper_engine.as_deref(), &mut *vad_engine, &output_path).await {
                            Ok((transcription, path)) => TranscriptionResult {
                                input: input.clone(),
                                transcription: Some(transcription),
                                path: path.unwrap_or("".to_string()),
                                timestamp,
                                error: None,
                            },
                            Err(e) => {
                                error!("STT error for input {}: {:?}", input.device, e);
                                TranscriptionResult {
                                    input: input.clone(),
                                    transcription: None,
                                    path: "".to_string(),
                                    timestamp,
                                    error: Some(e.to_string()),
                                }
                            },
                        }
                    };

                    if output_sender.send(transcription_result).is_err() {
                        break;
                    }
                }
                else => break,
            }
        }
        // Cleanup code here (if needed)
    });

    Ok((input_sender, output_receiver, shutdown_flag))
}