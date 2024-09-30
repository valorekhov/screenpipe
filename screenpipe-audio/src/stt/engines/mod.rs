pub mod whisper;
mod deepgram;
mod restpipe;

pub use deepgram::DeepgramEngine;
use restpipe::RestPipeEngine;
use whisper::{CandleWhisperModel, WhisperEngine};
use tokio::sync::watch;

use std::{
    collections::HashMap, path::PathBuf, sync::Arc, time::{SystemTime, UNIX_EPOCH}
};

use anyhow::Result;
use log::{debug, error, info};
#[cfg(target_os = "macos")]
use objc::rc::autoreleasepool;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use crate::{
    stt::{perform_stt, SttEngine, SttErrorKind}, vad_engine::{SileroVad, VadEngine, VadEngineEnum, WebRtcVad}, AudioInput, AudioTranscriptionEngine, TranscriptionResult, WhisperModel,
};

use super::RecordingState;

pub fn initialize_stt_engines(
    local_model: Option<CandleWhisperModel>,
    api_url: Option<String>,
    api_headers: Option<String>,
    deepgram_api_key: Option<String>,
) -> Result<(Box<dyn SttEngine + Send + Sync>, Option<Box<dyn SttEngine + Send + Sync>>)> {
    let local_model_opt = local_model.clone();
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
        // TODO: File payload field has tobe configurable
        Box::new(RestPipeEngine::new(url.clone(), api_headers, Some("file".to_string()), Some(16000)))
    } else {
        let whisper_model = match local_model_opt.unwrap_or(CandleWhisperModel::Tiny) {
            CandleWhisperModel::Tiny => AudioTranscriptionEngine::WhisperTiny,
            _ => AudioTranscriptionEngine::WhisperDistilLargeV3,
        };
        Box::new(WhisperEngine::new(WhisperModel::new(Arc::new(whisper_model))?).expect("Could not create the WhisperEngine"))
    };

    let fallback_engine: Option<Box<dyn SttEngine + Send + Sync>> = if deepgram_api_key.is_some() || api_url.is_some() {
        if let Some(local_model) = local_model{
            let whisper_model = match local_model {
                CandleWhisperModel::Tiny => AudioTranscriptionEngine::WhisperTiny,
                _ => AudioTranscriptionEngine::WhisperDistilLargeV3,
            };
            Some(Box::new(WhisperEngine::new(WhisperModel::new(Arc::new(whisper_model))?).expect("Could not create the WhisperEngine")))
        } else {
            None
        }
    } else {
        None
    };

    Ok((primary_engine, fallback_engine))
}

pub fn create_comm_channel(
    primary_whisper_engine: Box<dyn SttEngine + Send + Sync>,
    fallback_whisper_engine: Option<Box<dyn SttEngine + Send + Sync>>,
    vad_engine: VadEngineEnum,
    output_path: &Option<PathBuf>,
) -> Result<(
    UnboundedSender<AudioInput>,
    UnboundedReceiver<TranscriptionResult>,
    watch::Sender<RecordingState>,
    watch::Receiver<RecordingState>
)> {
    let (input_sender, mut input_receiver): (
        UnboundedSender<AudioInput>,
        UnboundedReceiver<AudioInput>,
    ) = unbounded_channel();
    let (output_sender, output_receiver): (
        UnboundedSender<TranscriptionResult>,
        UnboundedReceiver<TranscriptionResult>,
    ) = unbounded_channel();

    let (state_tx, state_rx) = watch::channel(RecordingState::Initializing);

    let mut vad_engine: Box<dyn VadEngine + Send> = match vad_engine {
        VadEngineEnum::WebRtc => Box::new(WebRtcVad::new()),
        VadEngineEnum::Silero => Box::new(SileroVad::new()?),
    };

    let output_path = output_path.clone();
    let state_rx_clone = state_rx.clone();
    let state_tx_clone = state_tx.clone();

    tokio::spawn(async move {
        loop {
            if state_rx_clone.has_changed().unwrap_or(false) {
                let state = *state_rx_clone.borrow();
                match state {
                    RecordingState::Stopping => {
                        info!("Whisper channel shutting down");
                        break;
                    }
                    _ => {
                        debug!("Continuing processing with {:?}...", state);
                    }
                }
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
                                handle_stt(&input, &*primary_whisper_engine, fallback_whisper_engine.as_deref(), &mut *vad_engine, &output_path, timestamp, &state_tx_clone).await
                            })
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            unreachable!("This code should not be reached on non-macOS platforms")
                        }
                    } else {
                        handle_stt(&input, &*primary_whisper_engine, fallback_whisper_engine.as_deref(), &mut *vad_engine, &output_path, timestamp, &state_tx_clone).await
                    };

                    if output_sender.send(transcription_result).is_err() {
                        break;
                    }

                    // if RecordingState::RecordingFinished == *state_rx_clone.borrow() {
                    //     break;
                    // }
                }
                else => {
                    break
                },
            }
        }
    });

    Ok((input_sender, output_receiver, state_tx, state_rx))
}

async fn handle_stt(
    input: &AudioInput,
    primary_whisper_engine: &(dyn SttEngine + Send + Sync),
    fallback_whisper_engine: Option<&(dyn SttEngine + Send + Sync)>,
    vad_engine: &mut (dyn VadEngine + Send),
    output_path: &Option<PathBuf>,
    timestamp: u64,
    state_tx: &watch::Sender<RecordingState>,
) -> TranscriptionResult {
    match perform_stt(input, primary_whisper_engine, fallback_whisper_engine, vad_engine, output_path).await {
        Ok((transcription, path)) => TranscriptionResult {
            input: input.clone(),
            transcription: Some(transcription),
            path: path.unwrap_or("".to_string()),
            timestamp,
            error: None,
        },
        Err(e) => {
            if let Some(SttErrorKind::NoSpeech) = e.downcast_ref::<SttErrorKind>() {
                debug!("No speech detected for input {}: {:?}. Finishing recording", input.device, e);
                if let Err(send_err) = state_tx.send(RecordingState::RecordingFinished) {
                    error!("Failed to send RecordingState::Stopping: {:?}", send_err);
                }
            } else {
                error!("STT error for input {}: {:?}", input.device, e);
            }
            TranscriptionResult {
                input: input.clone(),
                transcription: None,
                path: "".to_string(),
                timestamp,
                error: Some(e.to_string()),
            }
        },
    }
}