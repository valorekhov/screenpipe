use anyhow::{anyhow, Result};
use clap::Parser;
use copypasta::ClipboardContext;
use copypasta::ClipboardProvider;
use log::debug;
use log::info;
use log::warn;
use screenpipe_audio::create_comm_channel;
use screenpipe_audio::default_input_device;
use screenpipe_audio::default_output_device;
use screenpipe_audio::list_audio_devices;
use screenpipe_audio::parse_audio_device;
use screenpipe_audio::record_and_transcribe;
use screenpipe_audio::stt::engines::initialize_stt_engines;
use screenpipe_audio::stt::RecordingState;
use screenpipe_audio::AudioDevice;
use screenpipe_audio::AudioInput;
use screenpipe_audio::TranscriptionResult;
use screenpipe_audio::VadEngineEnum;
use screenpipe_audio::stt::engines::whisper::CandleWhisperModel;
use tokio::sync::watch::Receiver;
use tokio::sync::watch::Sender;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use tokio::sync::mpsc::UnboundedReceiver;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(
        short,
        long,
        help = "Audio device name (can be specified multiple times)"
    )]
    audio_device: Vec<String>,

    #[clap(long, help = "List available audio devices")]
    list_audio_devices: bool,

    #[clap(long, help = "Local model to use", value_enum)]
    local_model: Option<CandleWhisperModel>,

    #[clap(long, help = "Deepgram API key")]
    deepgram_api_key: Option<String>,

    #[clap(long, help = "API URL", conflicts_with = "deepgram_api_key", default_value = "http://localhost:5000/inference")]
    api_url: Option<String>,

    #[clap(long, help = "API Headers in the `Name: Value;` format", conflicts_with = "deepgram_api_key")]
    api_headers: Option<String>,

    #[clap(short, long, help = "Enable verbose output", conflicts_with = "very_verbose")]
    verbose: bool,
    
    #[clap(short = 'D', long = "very-verbose", help = "Enable very verbose output", conflicts_with = "verbose")]
    very_verbose: bool,

    #[clap(long, help = "Output to clipboard")]
    clipboard: bool,

    #[clap(short, long, help = "Output to file", value_name = "FILE")]
    file: Option<PathBuf>,

    #[clap(long, help = "Recording output directory", value_name = "DIR")]
    dir: Option<PathBuf>,

    #[clap(short, long, help = "Duration in seconds to record", default_value = "6")]
    duration: Option<u32>,
}

fn print_devices(devices: &[AudioDevice]) {
    println!("Available audio devices:");
    for (_, device) in devices.iter().enumerate() {
        println!("  {}", device);
    }

    #[cfg(target_os = "macos")]
    println!("On macOS, it's not intuitive but output devices are your displays");
}

// ! usage - cargo run --bin screenpipe-audio -- --audio-device "Display 1 (output)"

#[tokio::main]
async fn main() -> Result<()> {
    use env_logger::Builder;
    use log::LevelFilter;

    let args = Args::parse();

    Builder::new()
        .filter(None, if args.very_verbose {
            LevelFilter::Debug
        } else if args.verbose{
            LevelFilter::Info
        } else {
            LevelFilter::Error
        })
        .filter_module("tokenizers", LevelFilter::Error)
        .init();

    let devices = list_audio_devices().await?;

    if args.list_audio_devices {
        print_devices(&devices);
        return Ok(());
    }

    if let Some(dir) = &args.dir {
        if !dir.exists() {
            return Err(anyhow!("The specified directory does not exist: {:?}", dir));
        }
    }
    
    let devices = if args.audio_device.is_empty() {
        vec![default_input_device()?, default_output_device().await?]
    } else {
        args.audio_device
            .iter()
            .map(|d| parse_audio_device(d))
            .collect::<Result<Vec<_>>>()?
    };

    if devices.is_empty() {
        return Err(anyhow!("No audio input devices found"));
    }

    let chunk_duration = Duration::from_secs(5);
    let output_path = args.dir.map(PathBuf::from);

    let (primary_engine, fallback_engine) = initialize_stt_engines(
        args.local_model,
        args.api_url,
        args.api_headers,
        args.deepgram_api_key,
    )?;

    let (whisper_sender, whisper_receiver, state_tx, state_rx) = create_comm_channel(
        primary_engine,
        fallback_engine,
        VadEngineEnum::WebRtc,
        &output_path,
    )?;

    // Spawn recording threads
    let recording_threads = spawn_recording_threads(devices, whisper_sender, state_tx.clone(), state_rx.clone(),  chunk_duration);
    wait_for_initialization(state_rx.clone()).await?;

    // Spawn keyboard listener task
    let kb_task_join_handle = start_keyboard_listener_task(state_tx.clone(), state_rx.clone());

    // Spawn duration task if duration is specified
    if let Some(duration) = args.duration {
        start_max_duration_task(state_tx.clone(), duration as u64);
    }
  
    // Start main transcription loop
    let transcription_buffer = run_transcription_loop(whisper_receiver, state_rx, state_tx).await?;

    shutdown_and_cleanup(recording_threads, kb_task_join_handle).await?;

    if args.clipboard && !transcription_buffer.is_empty() {
        let mut ctx: ClipboardContext = ClipboardContext::new().expect("Could not create clipboard manager");
        ctx.set_contents(transcription_buffer.trim().to_owned()).expect("Could not set clipboard contents");
        info!("Copied to clipboard: {}", transcription_buffer);
    }

    println!("<|transcription|>{}</|transcription|>", transcription_buffer);

    info!("Application ending");

    Ok(())
}

fn start_keyboard_listener_task(state_tx: Sender<RecordingState>, mut state_rx: Receiver<RecordingState>) -> JoinHandle<()> {
    use device_query::{DeviceQuery, DeviceState, Keycode};
    
    tokio::spawn(async move {        
        let mut last_keys: Vec<Keycode> = Vec::new();
        loop {
            tokio::select! {
                Ok(()) = state_rx.changed() => {
                    if *state_rx.borrow() == RecordingState::Stopping {
                        debug!("Keyboard listener received stopping signal");
                        break;
                    }
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                    let device_state = DeviceState::new();
                    let keys = device_state.get_keys();
                    let new_keys: Vec<_> = keys.iter().filter(|k| !last_keys.contains(k)).collect();

                    for key in &new_keys {
                        debug!("Keyboard listener received key: {:?}", key);
                        match key {
                            Keycode::Enter => {
                                state_tx.send(RecordingState::RecordingFinished).expect("Unable to update recording state");
                                return;
                            }
                            Keycode::Space => {
                                let current_state = state_rx.borrow().clone();
                                match current_state {
                                    RecordingState::Recording => state_tx.send(RecordingState::RecordingPaused).expect("Unable to set state to Paused"),
                                    RecordingState::RecordingPaused => state_tx.send(RecordingState::Recording).expect("Unable to set state to Recording"),
                                    _ => {}
                                }
                            }
                            _ => {}
                        }
                    }
                    last_keys = keys;
                }
            }
        }
        debug!("Exiting Keyboard listener task");
    })
}

fn start_max_duration_task(state_tx: Sender<RecordingState>, duration: u64) {
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(duration)).await;
        state_tx.send(RecordingState::RecordingFinished).expect("Unable to update recording state to Finished");
    });
}

fn spawn_recording_threads(
    devices: Vec<AudioDevice>,
    whisper_sender: tokio::sync::mpsc::UnboundedSender<AudioInput>,
    state_tx: watch::Sender<RecordingState>,
    state_rx: watch::Receiver<RecordingState>,
    chunk_duration: Duration,
) -> Vec<tokio::task::JoinHandle<Result<()>>> {
    devices
        .into_iter()
        .map(|device| {
            let device = Arc::new(device);
            let whisper_sender = whisper_sender.clone();
            let device_clone = Arc::clone(&device);
            let state_tx = state_tx.clone();
            let state_rx_clone = state_rx.clone();

            tokio::spawn(async move {
                state_tx.send(RecordingState::Recording)?;

                let device_clone_2 = Arc::clone(&device_clone);
                
                record_and_transcribe(
                    device_clone_2,
                    chunk_duration,
                    whisper_sender,
                    state_rx_clone
                ).await?;

                debug!("Finished with record_and_transcribe");

                //state_tx.send(RecordingState::RecordingFinished)?;
                Ok(())
            })
        })
        .collect()
}

async fn wait_for_initialization(mut rx: watch::Receiver<RecordingState>) -> Result<()> {
    while *rx.borrow() == RecordingState::Initializing {
        rx.changed().await?;
    }
    Ok(())
}

async fn run_transcription_loop(
    mut whisper_receiver: UnboundedReceiver<TranscriptionResult>,
    mut state_rx: watch::Receiver<RecordingState>,
    state_tx: watch::Sender<RecordingState>,
) -> Result<String> {
    let mut transcription_buffer = String::new();
    let mut consecutive_timeouts = 0;
    let max_consecutive_timeouts = 3;

    loop {
        tokio::select! {
            Some(result) = whisper_receiver.recv() => {
                info!("Transcription: {:?}", result.transcription);
                consecutive_timeouts = 0;
                match result.transcription {
                    Some(text) => {
                        if !transcription_buffer.is_empty() {
                            transcription_buffer.push(' ');
                        }
                        transcription_buffer.push_str(&text);
                        if RecordingState::RecordingFinished == *state_rx.borrow() {
                            debug!("Recording has finished and no transcriptions are available. Exit here.");
                            break;
                        }
                    },
                    None if RecordingState::RecordingFinished == *state_rx.borrow() => {
                        debug!("Recording has finished and no transcriptions are available. Exit here.");
                        break;
                    },
                    None => {}
                }
            }
            Ok(()) = state_rx.changed() => {
                let state = *state_rx.borrow();
                debug!("Current state: {:?}", state);
                if  state == RecordingState::Stopping {
                    break;
                }
            }
            else => {
                let state = *state_rx.borrow();
                debug!("Current State: {:?}", state);

                consecutive_timeouts += 1;
                info!("No transcriptions received");
                if consecutive_timeouts >= max_consecutive_timeouts {
                    info!("No transcriptions received for a while, stopping...");
                    break;
                }
            }
        }
    }
    state_tx.send(RecordingState::Stopping)?;
    drain_remaining_transcriptions(&mut whisper_receiver).await;

    Ok(transcription_buffer)
}

async fn drain_remaining_transcriptions(
    whisper_receiver: &mut UnboundedReceiver<TranscriptionResult>
) {
    debug!("Draining remaining transcriptions...");
    let drain_timeout = Duration::from_secs(10);
    let drain_start = std::time::Instant::now();

    while let Ok(Some(result)) = timeout(drain_timeout.saturating_sub(drain_start.elapsed()), whisper_receiver.recv()).await {
        debug!("Drained transcription for device: {}", result.input.device);
        if drain_start.elapsed() >= drain_timeout {
            warn!("Draining timed out");
            break;
        }
    }

    debug!("Finished draining transcriptions");
}

async fn shutdown_and_cleanup(
    recording_threads: Vec<tokio::task::JoinHandle<Result<()>>>,
    kb_task_join_handle: JoinHandle<()>
) -> Result<()> {

    for (i, thread) in recording_threads.into_iter().enumerate() {
        thread.await??;
        info!("Recording {} complete", i);
    }
    kb_task_join_handle.await?;
    Ok(())
}
