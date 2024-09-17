use std::{
    path::PathBuf,
    sync::{atomic::{AtomicBool, Ordering}, Arc},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use log::{debug, error, info};
#[cfg(target_os = "macos")]
use objc::rc::autoreleasepool;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

use candle_transformers::models::whisper::{self as m, audio, Config};

use crate::{
    stt::{self, perform_stt, SttEngine}, vad_engine::{SileroVad, VadEngine, VadEngineEnum, WebRtcVad}, AudioInput, AudioTranscriptionEngine, TranscriptionResult
};


mod whisper_engine;
mod whisper_model;
mod model;

pub use whisper_engine::WhisperEngine;
pub use whisper_model::WhisperModel;
pub use model::{Model, token_id};

use super::DeepgramEngine;

pub async fn create_whisper_channel(
    audio_transcription_engine: Arc<AudioTranscriptionEngine>,
    vad_engine: VadEngineEnum,
    deepgram_api_key: Option<String>,
    output_path: &PathBuf,
) -> Result<(
    UnboundedSender<AudioInput>,
    UnboundedReceiver<TranscriptionResult>,
    Arc<AtomicBool>, // Shutdown flag
)> {
    let whisper_model = WhisperModel::new(audio_transcription_engine.clone())?;
    let whisper_engine = WhisperEngine::new(
        whisper_model,
    )?;
    let deepgram_engine = deepgram_api_key.map(DeepgramEngine::new);

    let (primary_engine, secondary_engine): (Arc<dyn SttEngine>, Option<Arc<dyn SttEngine>>) = deepgram_engine
        .map(|deepgram| (Arc::new(deepgram) as Arc<dyn SttEngine>, Some(Arc::new(whisper_engine) as Arc<dyn SttEngine>)))
        .unwrap_or_else(|| (Arc::new(whisper_engine) as Arc<dyn SttEngine>, None));
    
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
                                match perform_stt(&input, primary_engine, secondary_engine, &mut *vad_engine, &output_path) {
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
                        match perform_stt(&input, primary_engine, secondary_engine, &mut *vad_engine, &output_path) {
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
