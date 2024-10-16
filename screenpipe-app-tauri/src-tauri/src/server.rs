use axum::{
    extract::State,
    http::{Method, StatusCode},
    Json, Router,
};
use tower_http::cors::{CorsLayer, Any};
use tower_http::trace::{TraceLayer, DefaultMakeSpan};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tauri::Emitter;
#[allow(unused_imports)]
use tauri_plugin_notification::NotificationExt;
use tokio::sync::mpsc;
use tracing::{error, info};
use http::header::HeaderValue;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct LogEntry {
    pipe_id: String,
    level: String,
    message: String,
    timestamp: String,
}

#[derive(Clone)]
pub struct ServerState {
    app_handle: tauri::AppHandle,
}

#[derive(Serialize, Deserialize, Debug)]
struct NotificationPayload {
    title: String,
    body: String,
}

#[derive(Serialize)]
struct ApiResponse {
    success: bool,
    message: String,
}

pub async fn run_server(app_handle: tauri::AppHandle, port: u16) {
    let state = ServerState { app_handle };

    let cors = CorsLayer::new()
        .allow_origin("*".parse::<HeaderValue>().unwrap())
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any)
        .allow_credentials(false);

    let app = Router::new()
        .route("/notify", axum::routing::post(send_notification))
        .route("/log", axum::routing::post(log_message))
        .layer(cors)
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().include_headers(true)),
        )
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    info!("Server listening on {}", addr);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn send_notification(
    State(state): State<ServerState>,
    Json(payload): Json<NotificationPayload>,
) -> Result<Json<ApiResponse>, (StatusCode, String)> {
    info!("Received notification request: {:?}", payload);
    match state.app_handle.emit("notification-requested", &payload) {
        Ok(e) => {
            info!("Notification sent: {:?}", e);
            Ok(Json(ApiResponse {
                success: true,
                message: "Notification sent successfully".to_string(),
            }))
        }
        Err(e) => {
            error!("Failed to send notification: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to send notification: {}", e),
            ))
        }
    }
}

async fn log_message(
    State(state): State<ServerState>,
    Json(log_entry): Json<LogEntry>,
) -> Result<Json<ApiResponse>, (StatusCode, String)> {
    match state.app_handle.emit("log-message", &log_entry) {
        Ok(e) => {
            info!("Log message sent: {:?}", e);
            Ok(Json(ApiResponse {
                success: true,
                message: "Log message sent successfully".to_string(),
            }))
        }
        Err(e) => {
            error!("Failed to send log message: {}", e);
            Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to send log message: {}", e),
            ))
        }
    }
}

pub fn spawn_server(app_handle: tauri::AppHandle, port: u16) -> mpsc::Sender<()> {
    let (tx, mut rx) = mpsc::channel(1);

    tokio::spawn(async move {
        tokio::select! {
            _ = run_server(app_handle, port) => {},
            _ = rx.recv() => {
                info!("Received shutdown signal for server");
            }
        }
    });

    tx
}

/*


curl -X POST http://localhost:11435/notify \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Notification", "body": "This is a test notification body"}'

  curl -X POST http://localhost:11435/notify \
  -H "Content-Type: application/json" \
  -d '{"title": "Special Characters: !@#$%^&*()", "body": "Testing with émojis 😀🎉"}'

  curl -X POST http://localhost:11435/notify \
  -H "Content-Type: application/json" \
  -d '{"title": "Long Notification", "body": "This is a much longer notification body to test how the system handles larger amounts of text. It might wrap or be truncated depending on the system'\''s limitations."}'

  curl -X POST http://localhost:11435/notify \
  -H "Content-Type: application/json" \
  -d '{"title": "Malformed JSON", "body": "This JSON is malformed}'

*/
