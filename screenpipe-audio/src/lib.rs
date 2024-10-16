mod core;
mod multilingual;
pub mod pcm_decode;
pub mod stt;
pub mod vad_engine;
pub mod encode;
pub use core::{
    default_input_device, default_output_device, list_audio_devices, parse_audio_device,
    record_and_transcribe, AudioDevice, AudioTranscriptionEngine, DeviceControl, DeviceType
};
pub use encode::encode_single_audio;
pub use pcm_decode::pcm_decode;
pub use stt::{perform_stt, AudioInput, TranscriptionResult};
pub use stt::engines::whisper::{create_whisper_channel, WhisperModel};
pub use stt::engines::create_comm_channel;
pub use vad_engine::VadEngineEnum;