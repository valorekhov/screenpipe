use env_logger::{Builder, Env};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use std::{sync::{atomic::{AtomicBool, Ordering}, Arc}, thread};
use tokio::time::{Duration, sleep};
use log::{info, debug, error, warn};

// Simplified AudioInput struct
struct AudioInput {
    data: Vec<f32>,
    device: String,
    sample_rate: u32,
    channels: u16,
}

struct TranscriptionResult {
    input: AudioInput,
    transcription: Option<String>,
    timestamp: u64,
    error: Option<String>,
}

async fn create_comm_channel() -> (
    UnboundedSender<AudioInput>,
    UnboundedReceiver<TranscriptionResult>,
    Arc<AtomicBool>,
) {
    info!("Creating communication channel");
    let (input_sender, mut input_receiver) : (
        UnboundedSender<AudioInput>,
        UnboundedReceiver<AudioInput>,
    ) = unbounded_channel();
    let (output_sender, output_receiver): (
        UnboundedSender<TranscriptionResult>,
        UnboundedReceiver<TranscriptionResult>,
    ) = unbounded_channel();
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let shutdown_flag_clone = shutdown_flag.clone();

    tokio::spawn(async move {
        info!("Starting communication channel loop");
        loop {
            if shutdown_flag_clone.load(Ordering::Relaxed) {
                info!("Channel shutting down");
                break;
            }
            debug!("Waiting for input from input_receiver");

            match input_receiver.recv().await {
                Some(input) => {
                    debug!("Received input from input_receiver for device: {}", input.device);
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .expect("Time went backwards")
                        .as_secs();

                    debug!("Performing STT for input {}", input.device);
                    sleep(Duration::from_millis(300)).await;
                    info!("!! Finished with the delay for input {}", input.device);

                    let transcription_result = TranscriptionResult {
                        input,
                        transcription: Some("Simulated transcription".to_string()),
                        timestamp,
                        error: None,
                    };

                    debug!("Sending transcription result for device: {}", transcription_result.input.device);
                    if output_sender.send(transcription_result).is_err() {
                        error!("Failed to send transcription result");
                        break;
                    }
                },
                None => {
                    info!("Input channel closed, exiting loop");
                    break;
                }
            }
        }
        info!("Communication channel loop ended");
    });

    info!("Communication channel created");
    (input_sender, output_receiver, shutdown_flag)
}

pub async fn record_and_transcribe(
    audio_device: Arc<String>, // Simplified from Arc<AudioDevice>
    duration: Duration,
    whisper_sender: UnboundedSender<AudioInput>,
    is_running: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let sample_rate = 44100; // Simulated sample rate
    let channels = 2; // Simulated channels
    debug!(
        "Audio device config: sample_rate={}, channels={}",
        sample_rate, channels
    );
    let start_time = std::time::Instant::now();

    let audio_data = Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let is_running_weak = Arc::downgrade(&is_running);
    let audio_data_clone = Arc::clone(&audio_data);

    // Spawn a thread to simulate audio recording
    let audio_handle = thread::spawn(move || {
        while is_running_weak.upgrade().map_or(false, |arc| arc.load(Ordering::Relaxed)) {
            // Simulate recording audio data
            {
                let mut audio_data = audio_data_clone.blocking_lock();
                audio_data.extend_from_slice(&vec![0.0f32; 1000]); // Simulated audio data
            }
            thread::sleep(Duration::from_millis(100));
        }
    });

    info!(
        "Recording {} for {} seconds",
        audio_device,
        duration.as_secs()
    );

    // Wait for the duration unless is_running is false
    while is_running.load(Ordering::Relaxed) {
        sleep(Duration::from_millis(100)).await;
        if start_time.elapsed() > duration {
            debug!("Recording duration reached");
            break;
        }
    }

    // Signal the recording thread to stop
    is_running.store(false, Ordering::Relaxed);

    // Wait for the native thread to finish
    if let Err(e) = audio_handle.join() {
        error!("Error joining audio thread: {:?}", e);
    }

    debug!("Sending audio to audio model");
    let data = audio_data.lock().await;
    debug!("Sending audio of length {} to audio model", data.len());
    if let Err(e) = whisper_sender.send(AudioInput {
        data: data.clone(),
        device: (*audio_device).clone(),
        sample_rate,
        channels,
    }) {
        error!("Failed to send audio to audio model: {}", e);
    }
    debug!("Sent audio to audio model");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    Builder::from_env(Env::default().default_filter_or("debug")).init();
    info!("Application started");

    let (whisper_sender, mut whisper_receiver, _) = create_comm_channel().await;
    
    let devices = vec![
        "device1".to_string(),
        //"device2".to_string(),
    ];

    info!("Devices: {:?}", devices);

    let chunk_duration = Duration::from_secs(5);

    info!("Spawning recording threads");
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

                info!("Starting recording thread for device: {}", device_clone_2);
                record_and_transcribe(
                    device_clone_2,
                    chunk_duration,
                    whisper_sender,
                    device_control_clone,
                )
            })
        })
        .collect();

    info!("All recording threads spawned");

    let mut consecutive_timeouts = 0;
    let max_consecutive_timeouts = 3;

    info!("Starting main reception loop");
    loop {
        match whisper_receiver.try_recv() {
            Ok(result) => {
                info!("Received transcription for device: {}", result.input.device);
                debug!("Transcription: {:?}", result.transcription);
                consecutive_timeouts = 0;
            }
            Err(e) => {
                warn!("Error receiving transcription: {:?}", e);
                consecutive_timeouts += 1;
                if consecutive_timeouts >= max_consecutive_timeouts {
                    info!("No transcriptions received for a while, stopping...");
                    break;
                }
                debug!("Sleeping for 100ms before next receive attempt");
                sleep(Duration::from_millis(100)).await;
            }
        }
    }

    // In this state we are getting the reproduced behaviour with the STT task shutting down right after recording threads have finished
    // Uncomment the below to fix the issue

    // sleep(Duration::from_millis(1000)).await;
    // info!("Waiting for recording threads to complete");
    // for (i, thread) in recording_threads.into_iter().enumerate() {
    //     info!("Waiting for recording thread {}", i);
    //     thread.await.unwrap().await.unwrap();
    //     info!("Recording {} complete", i);
    // }

    info!("Application ending");
    Ok(())
}