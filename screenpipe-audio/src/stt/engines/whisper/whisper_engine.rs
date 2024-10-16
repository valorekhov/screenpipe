
use std::{future::Future, pin::Pin};

use anyhow::Result;
use candle::Tensor;
use log::debug;
#[cfg(target_os = "macos")]
use objc::rc::autoreleasepool;

use candle_transformers::models::whisper::audio;

use crate::{
    multilingual, stt::{engines::whisper::model::Decoder, SttEngine, Task}
};


use super::WhisperModel;

pub struct WhisperEngine {
    whisper_model: WhisperModel,
    mel_filters: Vec<f32>,
}

impl WhisperEngine {
    pub fn new(whisper_model: WhisperModel) -> Result<Self, anyhow::Error> {
        let model = &whisper_model.model;
        
        debug!("Loading mel filters");
        let mel_bytes = match model.config().num_mel_bins {
            80 => include_bytes!("../../../../models/whisper/melfilters.bytes").as_slice(),
            128 => include_bytes!("../../../../models/whisper/melfilters128.bytes").as_slice(),
            nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

        Ok(Self {
            whisper_model,
            mel_filters,
        })
    }
}

impl SttEngine for WhisperEngine {
    fn transcribe<'a>(
        &'a self,
        audio_data: &'a [f32],
        _sample_rate: u32,
        _channels: u16,
        device_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>> {
        Box::pin(async move {
            let model = &self.whisper_model.model;
            let tokenizer = &self.whisper_model.tokenizer;
            let device = &self.whisper_model.device;


            debug!("device: {}, converting pcm to mel spectrogram", device_name);
            let mel = audio::pcm_to_mel(model.config(), audio_data, &self.mel_filters);
            let mel_len = mel.len();
            debug!("device: {}, creating tensor from mel spectrogram", device_name);
            let mel = Tensor::from_vec(
                mel,
                (
                    1,
                    model.config().num_mel_bins,
                    mel_len / model.config().num_mel_bins,
                ),
                device,
            )?;

            debug!("device: {}, detecting language", device_name);
            let language_token = Some(multilingual::detect_language(
                &mut model.clone(),
                tokenizer,
                &mel,
            )?);
            let mut model = model.clone();
            debug!("device: {}, initializing decoder", device_name);
            let mut dc = Decoder::new(
                &mut model,
                tokenizer,
                42,
                device,
                language_token,
                Some(Task::Transcribe),
                true,
                false,
            )?;
            debug!("device: {}, starting decoding process", device_name);
            let segments = dc.run(&mel)?;
            debug!("device: {}, decoding complete", device_name);
            Ok(segments
                .iter()
                .map(|s| s.dr.text.clone())
                .collect::<Vec<String>>()
                .join("\n"))
        })
    }
}