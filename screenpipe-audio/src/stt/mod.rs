use std::{future::Future, io::Cursor, path::PathBuf, pin::Pin};

use thiserror::Error;
use anyhow::{Result, anyhow};
use chrono::Utc;
use cpal::SampleFormat;
use hound::{WavSpec, WavWriter};
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


pub mod engines;
pub trait SttEngine {
    fn transcribe<'a>(&'a self, audio_data: &'a [f32], sample_rate: u32, channels: u16, device_name: &'a str) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;
}

#[derive(Error, Debug)]
enum SttErrorKind {
    #[error("No speech detected in the audio")]
    NoSpeech,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Task {
    Transcribe,
    #[allow(dead_code)]
    Translate,
}

// Main STT function
pub async fn perform_stt(
    audio_input: &AudioInput,
    primary_engine: &(dyn SttEngine + Send + Sync),
    fallback_engine: Option<&(dyn SttEngine + Send + Sync)>,
    vad_engine: &mut (dyn VadEngine + Send),
    output_path: &Option<PathBuf>,
) -> Result<(String, Option<String>)> {
    let mut audio_data = audio_input.data.clone();
    //save_wav(PathBuf::from("/tmp/incoming_audio_f32.wav"), &mut audio_data, audio_input.sample_rate, audio_input.channels, SampleFormat::F32)?;
    
    let sample_rate = m::SAMPLE_RATE as u32;
    let mut new_channels = audio_input.channels;
    if audio_input.sample_rate != sample_rate {
        info!(
            "device: {}, resampling from {} Hz to {} Hz",
            audio_input.device,
            audio_input.sample_rate,
            sample_rate
        );
        audio_data = resample(audio_data, audio_input.channels, audio_input.sample_rate, sample_rate)?;
        info!("device: {}, resampling complete. Resampled into {} samples", audio_input.device, audio_data.len());
        new_channels = 1; // the resampled audio is mono
    }
    //save_wav(PathBuf::from("/tmp/resampled_audio_s16.wav"), &mut audio_data, sample_rate, new_channels, SampleFormat::I16)?;

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
        return Err(anyhow!("No speech detected in the audio").context(SttErrorKind::NoSpeech));
    }

    debug!(
        "device: {}, using {} speech frames out of {} total frames",
        audio_input.device,
        speech_frames.len() / frame_size,
        audio_data.len() / frame_size
    );

    //save_wav(PathBuf::from("/tmp/vad_audio_s16.wav"), &mut speech_frames, sample_rate, new_channels, SampleFormat::I16)?;
    let transcription = match primary_engine.transcribe(&speech_frames, sample_rate, new_channels, &audio_input.device).await {
        Ok(result) => result,
        Err(e) if fallback_engine.is_some() => {
            warn!(
                "device: {}, primary engine failed, falling back: {:?}",
                audio_input.device, e
            );
            fallback_engine.as_ref().unwrap().transcribe(&speech_frames, sample_rate, new_channels,&audio_input.device).await?
        } 
        Err(e) => return Err(anyhow::anyhow!("Primary engine failed and no fallback configured: {:?}", e)),
    };

    debug!("device: {}, transcription: {}", audio_input.device, transcription);

    let new_file_name = Utc::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let sanitized_device_name = audio_input.device.to_string().replace([' ', ':', '/', '\\'], "_");
    let file_path_clone = if let Some(output_path) = output_path {
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
        debug!("Saved transcription to {}", file_path_clone);
        Some(file_path_clone)
    } else {
        None
    };

    Ok((transcription, file_path_clone))
}

fn get_wav_format(sample_format: SampleFormat) -> Result<(u16, hound::SampleFormat)> {
    match sample_format {
        SampleFormat::I16 => Ok((16, hound::SampleFormat::Int)),
        SampleFormat::F32 => Ok((32, hound::SampleFormat::Float)),
        _ => Err(anyhow::anyhow!("Unsupported sample format")),
    }
}

fn create_wav(audio_data: &[f32], sample_rate: u32, channels: u16, sample_format: SampleFormat) -> Result<Vec<u8>> {
    let (bits_per_sample, wav_sample_format) = get_wav_format(sample_format)?;

    let spec: WavSpec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        sample_format: wav_sample_format,
    };
    
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec)?;
        write_samples(&mut writer, audio_data, wav_sample_format)?;
        writer.finalize()?;
    }

    Ok(cursor.into_inner())
}

fn save_wav(file_path: PathBuf, audio_data: &[f32], sample_rate: u32, channels: u16, sample_format: SampleFormat) -> Result<()> {
    let (bits_per_sample, wav_sample_format) = get_wav_format(sample_format)?;

    let spec: WavSpec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample,
        sample_format: wav_sample_format,
    };
    
    let file = std::fs::File::create(file_path)?;
    let mut writer = WavWriter::new(file, spec)?;

    write_samples(&mut writer, audio_data, wav_sample_format)?;

    writer.finalize()?;

    Ok(())
}

fn write_samples<W: std::io::Write + std::io::Seek>(
    writer: &mut WavWriter<W>,
audio_data: &[f32],
    wav_sample_format: hound::SampleFormat,
) -> Result<()> {
    if wav_sample_format == hound::SampleFormat::Int {
        let resampled_audio_i16: Vec<i16> = audio_data
            .iter()
            .map(|&sample| (sample * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect();

        for sample in resampled_audio_i16 {
            writer.write_sample(sample)?;
        }
    } else {
        for &sample in audio_data {
            writer.write_sample(sample)?;
        }
    }

    Ok(())
}

fn resample(
    input: Vec<f32>,
    input_channels: u16,
    from_sample_rate: u32,
    to_sample_rate: u32
) -> Result<Vec<f32>> {
    debug!("Resampling audio: {} -> {}, {} len", from_sample_rate, to_sample_rate, input.len());
    
    // Resampler parameters
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    // If we have multiple channels, mix them down into mono by averaging interleaved data
    let mono_input: Vec<f32> = if input_channels > 1 {
        debug!("Mixing down to mono");
        input
            .chunks(input_channels as usize)  // Each chunk represents a frame (one sample per channel)
            .map(|frame| {
                // Average the channels to create mono
                (frame.iter().map(|&x| x as f64).sum::<f64>() / input_channels as f64) as f32
            })
            .collect()
    } else {
        input  
    };

    // Set up the resampler
    let mut resampler = SincFixedIn::<f32>::new(
        to_sample_rate as f64 / from_sample_rate as f64, // Resampling ratio
        2.0,       // Maximum relative output size
        params,                     // Interpolation parameters
        mono_input.len(),           // Number of samples in input
        1,                        // Number of channels in output (mono)
    )?;

    debug!("Performing resampling: {} len", mono_input.len());
    let waves_in = vec![mono_input];
    let waves_out = resampler.process(&waves_in, None)?;

    debug!("Resampling complete: {} len", waves_out[0].len());
    // Return the resampled data, assuming only one output channel
    if let Some(resampled) = waves_out.into_iter().next() {
        Ok(resampled)
    } else {
        Err(anyhow::Error::msg("No resampled data produced"))
    }
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

#[derive(Clone, PartialEq, Debug, Copy)]
pub enum RecordingState {
    Initializing,
    Recording,
    RecordingPaused,
    RecordingFinished,
    Stopping,
    Draining,
}
