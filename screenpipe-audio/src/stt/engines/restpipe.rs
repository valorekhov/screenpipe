
async fn transcribe_with_restpipe(
    wav_file: &[u8],
    device: &str,
    api_url: &str,
) -> Result<String, anyhow::Error> {
    let client = reqwest::Client::new();
    
    debug!("Sending request to RestPipe API");
    let response = client
        .post(api_url)
        .body(wav_file.to_vec())
        .header("Content-Type", "audio/wav")
        .send()
        .await?;

    match response.status() {
        reqwest::StatusCode::OK => {
            debug!("Received successful response from RestPipe API");
            let transcription = response.text().await?;
            
            if transcription.is_empty() {
                info!(
                    "device: {}, transcription is empty",
                    device
                );
            } else {
                info!(
                    "device: {}, transcription successful. length: {} characters",
                    device,
                    transcription.len()
                );
            }

            Ok(transcription)
        }
        status => {
            let error_message = format!("RestPipe API error: HTTP {}", status);
            error!("{}", error_message);
            Err(anyhow::anyhow!(error_message))
        }
    }
}

