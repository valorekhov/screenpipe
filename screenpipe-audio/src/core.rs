use crate::stt::RecordingState;
use crate::AudioInput;
use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::{debug, error, info, warn};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use std::{fmt, thread};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::{watch, Mutex};

#[derive(Clone, Debug, PartialEq)]
#[derive(Default)]
pub enum AudioTranscriptionEngine {
    RestPipe,
    Deepgram,
    #[default]
    WhisperTiny,
    WhisperDistilLargeV3,
}

impl fmt::Display for AudioTranscriptionEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AudioTranscriptionEngine::RestPipe => write!(f, "RestPipe"),
            AudioTranscriptionEngine::Deepgram => write!(f, "Deepgram"),
            AudioTranscriptionEngine::WhisperTiny => write!(f, "WhisperTiny"),
            AudioTranscriptionEngine::WhisperDistilLargeV3 => write!(f, "WhisperLarge"),
        }
    }
}


#[derive(Clone)]
pub struct DeviceControl {
    pub is_running: bool,
    pub is_paused: bool,
}

#[derive(Clone, Eq, PartialEq, Hash, Serialize)]
pub enum DeviceType {
    Input,
    Output,
}

#[derive(Clone, Eq, PartialEq, Hash, Serialize)]
pub struct AudioDevice {
    pub name: String,
    pub device_type: DeviceType,
}

impl AudioDevice {
    pub fn new(name: String, device_type: DeviceType) -> Self {
        AudioDevice { name, device_type }
    }

    pub fn from_name(name: &str) -> Result<Self> {
        if name.trim().is_empty() {
            return Err(anyhow!("Device name cannot be empty"));
        }

        let (name, device_type) = if name.to_lowercase().ends_with("(input)") {
            (
                name.trim_end_matches("(input)").trim().to_string(),
                DeviceType::Input,
            )
        } else if name.to_lowercase().ends_with("(output)") {
            (
                name.trim_end_matches("(output)").trim().to_string(),
                DeviceType::Output,
            )
        } else {
            return Err(anyhow!(
                "Device type (input/output) not specified in the name"
            ));
        };

        Ok(AudioDevice::new(name, device_type))
    }
}

impl fmt::Display for AudioDevice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} ({})",
            self.name,
            match self.device_type {
                DeviceType::Input => "input",
                DeviceType::Output => "output",
            }
        )
    }
}

pub fn parse_audio_device(name: &str) -> Result<AudioDevice> {
    AudioDevice::from_name(name)
}

async fn get_device_and_config(
    audio_device: &AudioDevice,
) -> Result<(cpal::Device, cpal::SupportedStreamConfig)> {
    let host = cpal::default_host();

    info!("device: {:?}", audio_device.to_string());

    let is_output_device = audio_device.device_type == DeviceType::Output;
    let is_display = audio_device.to_string().contains("Display");

    let cpal_audio_device = if audio_device.to_string() == "default" {
        match audio_device.device_type {
            DeviceType::Input => host.default_input_device(),
            DeviceType::Output => host.default_output_device(),
        }
    } else {
        let mut devices = match audio_device.device_type {
            DeviceType::Input => host.input_devices()?,
            DeviceType::Output => host.output_devices()?,
        };

        #[cfg(target_os = "macos")]
        {
            if audio_device.device_type == DeviceType::Output {
                if let Ok(screen_capture_host) = cpal::host_from_id(cpal::HostId::ScreenCaptureKit)
                {
                    devices = screen_capture_host.input_devices()?;
                }
            }
        }

        devices.find(|x| {
            x.name()
                .map(|y| {
                    y == audio_device
                        .to_string()
                        .replace(" (input)", "")
                        .replace(" (output)", "")
                        .trim()
                })
                .unwrap_or(false)
        })
    }
    .ok_or_else(|| anyhow!("Audio device not found"))?;

    // if output device and windows, using output config
    let config = if is_output_device && !is_display {
        cpal_audio_device.default_output_config()?
    } else {
        cpal_audio_device.default_input_config()?
    };
    Ok((cpal_audio_device, config))
}

pub async fn record_and_transcribe(
    audio_device: Arc<AudioDevice>,
    chunk_duration: Duration,
    whisper_sender: UnboundedSender<AudioInput>,
    state_rx: watch::Receiver<RecordingState>,
) -> Result<()> {
    let (cpal_audio_device, config) = get_device_and_config(&audio_device).await?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels();
    debug!(
        "Audio device config: sample_rate={}, channels={}",
        sample_rate, channels
    );

    let audio_data = Arc::new(Mutex::new(Vec::new()));
    let is_running = Arc::new(AtomicBool::new(true));
    let is_running_clone = is_running.clone();
    let is_running_weak_2 = Arc::downgrade(&is_running);

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let whisper_sender = Arc::new(whisper_sender);

    fn build_stream<T>(
        device: &cpal::Device,
        config: cpal::StreamConfig,
        audio_data: Arc<Mutex<Vec<f32>>>,
        tx: UnboundedSender<Vec<f32>>,
        is_running: Arc<AtomicBool>,
        chunk_duration: Duration
    ) -> Result<cpal::Stream>
    where
        T: cpal::Sample + cpal::SizedSample + bytemuck::Pod,
    {
        let is_running_weak = Arc::downgrade(&is_running);
        device.build_input_stream(
            &config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                if !is_running.load(Ordering::Relaxed) {
                    return;
                }

                // Convert the input data to f32
                let samples: &[f32] = bytemuck::cast_slice(data);

                let mut buffer = audio_data.blocking_lock();
                buffer.extend_from_slice(samples);

                // Check if the buffer has reached the chunk size
                if buffer.len() >= (chunk_duration.as_secs() as usize) * config.sample_rate.0 as usize * config.channels as usize {
                    let chunk = buffer.split_off(0); // Remove the chunk from the buffer
                    tx.send(chunk).ok();
                }
            },
            move |err| {
                error!("An error occurred on the audio stream: {}", err);
                if err.to_string().contains("device is no longer valid") {
                    warn!("Audio device disconnected. Stopping recording.");
                    if let Some(arc) = is_running_weak.upgrade() {
                        arc.store(false, Ordering::Relaxed);
                    }
                }
            },
            None
        )
        .map_err(|e| anyhow::anyhow!(e))
    }

    let audio_handle = thread::spawn(move || {
        let stream = match config.sample_format() {
            cpal::SampleFormat::I8 => build_stream::<i8>(&cpal_audio_device, config.into(), Arc::clone(&audio_data), tx.clone(), Arc::clone(&is_running), chunk_duration),
            cpal::SampleFormat::I16 => build_stream::<i16>(&cpal_audio_device, config.into(), Arc::clone(&audio_data), tx.clone(), Arc::clone(&is_running), chunk_duration),
            cpal::SampleFormat::I32 => build_stream::<i32>(&cpal_audio_device, config.into(), Arc::clone(&audio_data), tx.clone(), Arc::clone(&is_running), chunk_duration),
            cpal::SampleFormat::F32 => build_stream::<f32>(&cpal_audio_device, config.into(), Arc::clone(&audio_data), tx.clone(), Arc::clone(&is_running), chunk_duration),
            _ => {
                error!("Unsupported sample format: {:?}", config.sample_format());
                return;
            }
        };

        match stream {
            Ok(s) => {
                if let Err(e) = s.play() {
                    error!("Failed to play stream: {}", e);
                }
                // Keep the stream alive until the recording is done
                while is_running_weak_2
                    .upgrade()
                    .map_or(false, |arc| arc.load(Ordering::Relaxed))
                {
                    std::thread::sleep(Duration::from_millis(100));
                }
                s.pause().ok();
                drop(s);
            }
            Err(e) => error!("Failed to build input stream: {}", e),
        }
    });

    info!(
        "Recording {} continuously",
        audio_device.to_string(),
    );

    while *state_rx.borrow() == RecordingState::Recording {
        if let Some(chunk) = rx.recv().await {
            debug!("Sending audio chunk of length {} to audio model", chunk.len());
            if let Err(e) = whisper_sender.send(AudioInput {
                data: chunk,
                device: audio_device.to_string(),
                sample_rate,
                channels,
            }) {
                error!("Failed to send audio to audio model: {}", e);
            }
            debug!("Sent audio chunk to audio model");
        }
    }

    //Signal the recording thread to stop
    is_running_clone.store(false, Ordering::Relaxed);

    // Wait for the native thread to finish
    if let Err(e) = audio_handle.join() {
        error!("Error joining audio thread: {:?}", e);
    }

    Ok(())
}
pub async fn list_audio_devices() -> Result<Vec<AudioDevice>> {
    let host = cpal::default_host();
    let mut devices = Vec::new();

    for device in host.input_devices()? {
        if let Ok(name) = device.name() {
            devices.push(AudioDevice::new(name, DeviceType::Input));
        }
    }

    // Filter function to exclude macOS speakers and AirPods for output devices
    fn should_include_output_device(name: &str) -> bool {
        !name.to_lowercase().contains("speakers") && !name.to_lowercase().contains("airpods")
    }

    // macos hack using screen capture kit for output devices - does not work well
    #[cfg(target_os = "macos")]
    {
        // !HACK macos is suppoed to use special macos feature "display capture"
        // ! see https://github.com/RustAudio/cpal/pull/894
        if let Ok(host) = cpal::host_from_id(cpal::HostId::ScreenCaptureKit) {
            for device in host.input_devices()? {
                if let Ok(name) = device.name() {
                    if should_include_output_device(&name) {
                        devices.push(AudioDevice::new(name, DeviceType::Output));
                    }
                }
            }
        }
    }

    // add default output device - on macos think of custom virtual devices
    for device in host.output_devices()? {
        if let Ok(name) = device.name() {
            if should_include_output_device(&name) {
                devices.push(AudioDevice::new(name, DeviceType::Output));
            }
        }
    }

    Ok(devices)
}

pub fn default_input_device() -> Result<AudioDevice> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or(anyhow!("No default input device detected"))?;
    Ok(AudioDevice::new(device.name()?, DeviceType::Input))
}
// this should be optional ?
pub async fn default_output_device() -> Result<AudioDevice> {
    #[cfg(target_os = "macos")]
    {
        // ! see https://github.com/RustAudio/cpal/pull/894
        if let Ok(host) = cpal::host_from_id(cpal::HostId::ScreenCaptureKit) {
            if let Some(device) = host.default_input_device() {
                if let Ok(name) = device.name() {
                    return Ok(AudioDevice::new(name, DeviceType::Output));
                }
            }
        }
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("No default output device found"))?;
        return Ok(AudioDevice::new(device.name()?, DeviceType::Output));
    }

    #[cfg(not(target_os = "macos"))]
    {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("No default output device found"))?;
        Ok(AudioDevice::new(device.name()?, DeviceType::Output))
    }
}
