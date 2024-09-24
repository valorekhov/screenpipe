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
use screenpipe_audio::AudioDevice;
use screenpipe_audio::VadEngineEnum;
use screenpipe_audio::stt::engines::whisper::CandleWhisperModel;
use tokio::time::timeout;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

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

    #[clap(long, help = "API URL", conflicts_with = "deepgram_api_key")]
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

    // delete .mp4 files (output*.mp4)
    //std::fs::remove_file("output_0.mp4").unwrap_or_default();
    //std::fs::remove_file("output_1.mp4").unwrap_or_default();

    let chunk_duration = Duration::from_secs(5);
    let output_path = args.dir.map(PathBuf::from);

    let (primary_engine, fallback_engine) = initialize_stt_engines(
        args.local_model,
        args.api_url,
        args.api_headers,
        args.deepgram_api_key,
    )?;

    let (whisper_sender, mut whisper_receiver, shutdown_flag) = create_comm_channel(
        primary_engine,
        fallback_engine,
        VadEngineEnum::WebRtc,
        &output_path,
    )?;
    // Spawn threads for each device
    let recording_threads: Vec<_> = devices
        .into_iter()
        .map(|device| {
            let device = Arc::new(device);
            let whisper_sender = whisper_sender.clone();
            let device_control = Arc::new(AtomicBool::new(true));
            let device_clone = Arc::clone(&device);

            tokio::spawn(async move {
                let device_control_clone = Arc::clone(&device_control);
                let device_clone_2 = Arc::clone(&device_clone);

                record_and_transcribe(
                    device_clone_2,
                    chunk_duration,
                    whisper_sender,
                    device_control_clone,
                )
            })
        })
        .collect();
    let mut consecutive_timeouts = 0;
    let max_consecutive_timeouts = 3; // Adjust this value as needed

    // Main loop to receive and print transcriptions
    let mut transcription_buffer = String::new();

    loop {
        match whisper_receiver.try_recv() {
            Ok(result) => {
                info!("Transcription: {:?}", result);
                consecutive_timeouts = 0; // Reset the counter on successful receive
                if let Some(text) = result.transcription {
                    transcription_buffer.push_str(&text);
                    transcription_buffer.push(' '); // Add space between chunks
                }
            }
            Err(_) => {
                consecutive_timeouts += 1;
                if consecutive_timeouts >= max_consecutive_timeouts {
                    info!("No transcriptions received for a while, stopping...");
                    break;
                }
                continue;
            }
        }
    }

    // Wait for all recording threads to finish
    for (i, thread) in recording_threads.into_iter().enumerate() {
        let file_path = thread.await.unwrap().await;
        info!("Recording {} complete: {:?}", i, file_path);
    }

    // Shutdown the whisper_receiver
    shutdown_flag.store(true, Ordering::Relaxed);

    // Drain the whisper_receiver
    debug!("Draining remaining transcriptions...");
    let drain_timeout = Duration::from_secs(10); // Adjust as needed
    let drain_start = std::time::Instant::now();

    while let Ok(Some(result)) = timeout(drain_timeout.saturating_sub(drain_start.elapsed()), whisper_receiver.recv()).await {
        debug!("Drained transcription for device: {}", result.input.device);
        // if let Some(text) = result.transcription {
        //     transcription_buffer.push_str(&text);
        //     transcription_buffer.push(' ');
        // }
        if drain_start.elapsed() >= drain_timeout {
            warn!("Draining timed out");
            break;
        }
    }

    if args.clipboard && !transcription_buffer.is_empty() {
        let mut ctx: ClipboardContext = ClipboardContext::new().unwrap();
        ctx.set_contents(transcription_buffer.trim().to_owned()).unwrap();
        debug!("Copied to clipboard: {}", transcription_buffer);
    }

    println!("{}", transcription_buffer);

    debug!("Finished draining transcriptions");
    info!("Application ending");

    Ok(())
}
