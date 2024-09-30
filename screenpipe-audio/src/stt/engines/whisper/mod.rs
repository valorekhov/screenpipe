use std::{
    path::PathBuf,
    sync::{atomic::{AtomicBool, Ordering}, Arc},
};

use anyhow::Result;
#[cfg(target_os = "macos")]
use objc::rc::autoreleasepool;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};


use crate::{
    stt::RecordingState, vad_engine::VadEngineEnum, AudioInput, AudioTranscriptionEngine, TranscriptionResult
};


mod whisper_engine;
mod whisper_model;
mod model;

pub use whisper_engine::WhisperEngine;
pub use whisper_model::WhisperModel;
pub use model::{Model, token_id};

use super::create_comm_channel;

#[derive(clap::ValueEnum, Clone, Debug, PartialEq)]
pub enum CandleWhisperModel {
    Tiny,
    DistillLarge,
}

pub fn create_whisper_channel(
    audio_transcription_engine: Arc<AudioTranscriptionEngine>,
    vad_engine: VadEngineEnum,
    deepgram_api_key: Option<String>,
    output_path: &PathBuf,
) -> Result<(
    UnboundedSender<AudioInput>,
    UnboundedReceiver<TranscriptionResult>,
    Arc<AtomicBool>, // Shutdown flag
)> {
    let (primary_engine, fallback_engine) = super::initialize_stt_engines(
        match (*audio_transcription_engine).clone() {
            AudioTranscriptionEngine::WhisperTiny => Some(CandleWhisperModel::Tiny),
            AudioTranscriptionEngine::WhisperDistilLargeV3 => Some(CandleWhisperModel::DistillLarge),
            _ => None,
        },
        None,
        None,
        deepgram_api_key,
    ).expect("Failed to initialize engines");

    let (sender, receiver, _, state_rx) = create_comm_channel(primary_engine, fallback_engine, vad_engine, &Some(output_path.to_owned()))?;

    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let shutdown_flag_clone = Arc::clone(&shutdown_flag);
    let mut state_rx = state_rx.clone();
    tokio::spawn(async move {
        while let Ok(_) = state_rx.changed().await {
            let state = *state_rx.borrow();
            match state {
                RecordingState::RecordingFinished | RecordingState::Stopping => {
                    shutdown_flag_clone.store(true, Ordering::SeqCst);
                    break;
                }
                _ => {}
            }
        }
    });
    
    Ok((sender, receiver, shutdown_flag))
}