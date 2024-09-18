use std::path::PathBuf;

use anyhow::Result;
use chrono::Utc;
use log::{debug, info, warn};
#[cfg(target_os = "macos")]
use objc::rc::autoreleasepool;

use candle_transformers::models::whisper::{self as m};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use crate::{
    encode_single_audio,
    vad_engine::VadEngine,
};


pub(crate) mod engines;
pub trait SttEngine {
    fn transcribe(&self, audio_data: &[f32], sample_rate: u32, device_name: &str) -> Result<String>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Task {
    Transcribe,
    #[allow(dead_code)]
    Translate,
}

// Main STT function
pub fn perform_stt(
    audio_input: &AudioInput,
    primary_engine: &Box<dyn SttEngine + Send>,
    fallback_engine: &Option<Box<dyn SttEngine + Send>>,
    vad_engine: &mut dyn VadEngine,
    output_path: &PathBuf,
) -> Result<(String, String)> {
    let mut audio_data = audio_input.data.clone();
    if audio_input.sample_rate != m::SAMPLE_RATE as u32 {
        info!(
            "device: {}, resampling from {} Hz to {} Hz",
            audio_input.device,
            audio_input.sample_rate,
            m::SAMPLE_RATE
        );
        audio_data = resample(audio_data, audio_input.sample_rate, m::SAMPLE_RATE as u32)?;
    }

    // Filter out non-speech segments using Silero VAD
    debug!(
        "device: {}, filtering out non-speech segments with VAD",
        audio_input.device
    );
    let frame_size = 160; // 10ms frame size for 16kHz audio
    let mut speech_frames = Vec::new();
    for (frame_index, chunk) in audio_data.chunks(frame_size).enumerate() {
        match vad_engine.is_voice_segment(chunk) {
            Ok(is_voice) => {
                if is_voice {
                    speech_frames.extend_from_slice(chunk);
                }
            }
            Err(e) => {
                debug!("VAD failed for frame {}: {:?}", frame_index, e);
            }
        }
    }

    info!(
        "device: {}, total audio frames processed: {}, frames that include speech: {}",
        audio_input.device,
        audio_data.len() / frame_size,
        speech_frames.len() / frame_size
    );

    // If no speech frames detected, skip processing
    if speech_frames.is_empty() {
        debug!(
            "device: {}, no speech detected using VAD, skipping audio processing",
            audio_input.device
        );
        return Ok(("".to_string(), "".to_string())); // Return an empty string or consider a more specific "no speech" indicator
    }

    debug!(
        "device: {}, using {} speech frames out of {} total frames",
        audio_input.device,
        speech_frames.len() / frame_size,
        audio_data.len() / frame_size
    );

    let transcription = match primary_engine.transcribe(&speech_frames, audio_input.sample_rate, &audio_input.device) {
        Ok(result) => result,
        Err(e) if fallback_engine.is_some() => {
            warn!(
                "device: {}, primary engine failed, falling back: {:?}",
                audio_input.device, e
            );
            fallback_engine.as_ref().unwrap().transcribe(&speech_frames, audio_input.sample_rate, &audio_input.device)?
        }
        Err(e) => return Err(e),
    };

    let new_file_name = Utc::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let sanitized_device_name = audio_input.device.to_string().replace(['/', '\\'], "_");
    let file_path = PathBuf::from(output_path)
        .join(format!("{}_{}.mp4", sanitized_device_name, new_file_name))
        .to_str()
        .expect("Failed to create valid path")
        .to_string();
    debug!("Saving transcription to {}", file_path);
    let file_path_clone = file_path.clone();
    // Run FFmpeg in a separate task
    encode_single_audio(
        bytemuck::cast_slice(&audio_input.data),
        audio_input.sample_rate,
        audio_input.channels,
        &file_path.into(),
    )?;

    Ok((transcription, file_path_clone))
}

fn resample(input: Vec<f32>, from_sample_rate: u32, to_sample_rate: u32) -> Result<Vec<f32>> {
    debug!("Resampling audio");
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        to_sample_rate as f64 / from_sample_rate as f64,
        2.0,
        params,
        input.len(),
        1,
    )?;

    let waves_in = vec![input];
    debug!("Performing resampling");
    let waves_out = resampler.process(&waves_in, None)?;
    debug!("Resampling complete");
    Ok(waves_out.into_iter().next().unwrap())
}

#[derive(Debug, Clone)]
pub struct AudioInput {
    pub data: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub device: String,
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub path: String,
    pub input: AudioInput,
    pub transcription: Option<String>,
    pub timestamp: u64,
    pub error: Option<String>,
}

