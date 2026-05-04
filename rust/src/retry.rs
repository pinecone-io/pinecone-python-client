use std::collections::HashSet;
use std::future::Future;
use std::time::Duration;

use rand::Rng;
use tonic::Status;

/// Configuration for retry behavior on gRPC calls.
#[derive(Clone, Debug)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries, just the initial call).
    pub max_retries: u32,
    /// Initial backoff duration before the first retry.
    pub initial_backoff: Duration,
    /// Maximum backoff duration cap.
    pub max_backoff: Duration,
    /// Backoff multiplier applied each attempt.
    pub multiplier: u32,
    /// gRPC status codes that trigger a retry. Defaults to UNAVAILABLE, RESOURCE_EXHAUSTED,
    /// ABORTED — Pinecone data-plane operations (upsert, query, fetch, delete-by-id, update)
    /// are idempotent and safe to retry on these transient codes.
    pub retryable_codes: HashSet<i32>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(1600),
            multiplier: 2,
            retryable_codes: [
                tonic::Code::Unavailable as i32,
                tonic::Code::ResourceExhausted as i32,
                tonic::Code::Aborted as i32,
            ]
            .into_iter()
            .collect(),
        }
    }
}

/// Execute an async gRPC operation with retry on transient error codes.
///
/// Uses full-jitter exponential backoff:
///   delay = random(0, min(max_backoff, initial_backoff * multiplier^attempt))
///
/// Retries on any code listed in `config.retryable_codes` (default: UNAVAILABLE,
/// RESOURCE_EXHAUSTED, ABORTED). All other error codes are returned immediately without retry.
pub async fn retry_on_transient<F, Fut, T>(
    config: &RetryConfig,
    mut operation: F,
) -> Result<T, Status>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, Status>>,
{
    let mut attempt = 0u32;

    loop {
        match operation().await {
            Ok(val) => return Ok(val),
            Err(status) if config.retryable_codes.contains(&(status.code() as i32)) => {
                if attempt >= config.max_retries {
                    return Err(status);
                }
                let base = config
                    .initial_backoff
                    .saturating_mul(config.multiplier.saturating_pow(attempt));
                let capped = std::cmp::min(base, config.max_backoff);
                let jittered = if capped.is_zero() {
                    Duration::ZERO
                } else {
                    let ms = rand::rng().random_range(0..=capped.as_millis() as u64);
                    Duration::from_millis(ms)
                };
                // TODO: honor server-supplied pushback hints (grpc-retry-pushback-ms trailer
                // or Retry-After in trailers) before falling back to computed jitter backoff.
                tokio::time::sleep(jittered).await;
                attempt += 1;
            }
            Err(status) => return Err(status),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    fn test_config(max_retries: u32) -> RetryConfig {
        RetryConfig {
            max_retries,
            initial_backoff: Duration::from_millis(1), // fast for tests
            max_backoff: Duration::from_millis(10),
            multiplier: 2,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn retry_occurs_on_unavailable() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(5);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(Status::unavailable("service unavailable"))
                } else {
                    Ok::<&str, Status>("success")
                }
            }
        })
        .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "success");
        // Initial call + 2 retries = 3 total calls
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn no_retry_on_deadline_exceeded() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(5);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<(), Status>(Status::deadline_exceeded("timeout"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::DeadlineExceeded);
        // Only 1 call, no retries
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retry_occurs_on_resource_exhausted() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(5);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(Status::resource_exhausted("rate limited"))
                } else {
                    Ok::<(), Status>(())
                }
            }
        })
        .await;

        assert!(result.is_ok());
        // Initial call + 2 retries = 3 total calls
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retry_occurs_on_aborted() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(5);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(Status::aborted("conflict"))
                } else {
                    Ok::<(), Status>(())
                }
            }
        })
        .await;

        assert!(result.is_ok());
        // Initial call + 2 retries = 3 total calls
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn no_retry_on_internal() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(5);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<(), Status>(Status::internal("oops"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Internal);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn custom_retryable_codes_override() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = RetryConfig {
            max_retries: 3,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(10),
            multiplier: 2,
            retryable_codes: HashSet::from([tonic::Code::DeadlineExceeded as i32]),
        };
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                let n = count.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(Status::deadline_exceeded("timeout"))
                } else {
                    Ok::<(), Status>(())
                }
            }
        })
        .await;

        assert!(result.is_ok());
        // DEADLINE_EXCEEDED is retried under this custom config
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn respects_max_retry_count() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(3);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<(), Status>(Status::unavailable("always unavailable"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::Unavailable);
        // 1 initial + 3 retries = 4 total calls
        assert_eq!(call_count.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn zero_retries_means_no_retry() {
        let call_count = Arc::new(AtomicU32::new(0));
        let count = call_count.clone();

        let config = test_config(0);
        let result = retry_on_transient(&config, || {
            let count = count.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Err::<(), Status>(Status::unavailable("unavailable"))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn backoff_delay_increases() {
        // Verify that successive retries take progressively longer.
        // We measure wall-clock time for configs with different retry counts.
        let config_1 = RetryConfig {
            max_retries: 1,
            initial_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(500),
            multiplier: 2,
            ..Default::default()
        };
        let config_3 = RetryConfig {
            max_retries: 3,
            initial_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(500),
            multiplier: 2,
            ..Default::default()
        };

        let start_1 = std::time::Instant::now();
        let _ = retry_on_transient(&config_1, || async {
            Err::<(), Status>(Status::unavailable("unavailable"))
        })
        .await;
        let elapsed_1 = start_1.elapsed();

        let start_3 = std::time::Instant::now();
        let _ = retry_on_transient(&config_3, || async {
            Err::<(), Status>(Status::unavailable("unavailable"))
        })
        .await;
        let elapsed_3 = start_3.elapsed();

        // 3 retries should take longer than 1 retry due to increasing backoff
        assert!(
            elapsed_3 > elapsed_1,
            "3 retries ({elapsed_3:?}) should take longer than 1 retry ({elapsed_1:?})"
        );
    }

    #[tokio::test]
    async fn success_on_first_attempt_returns_immediately() {
        let config = test_config(5);
        let result =
            retry_on_transient(&config, || async { Ok::<&str, Status>("immediate") }).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "immediate");
    }
}
