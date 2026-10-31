//! The clock behind time limits and the elapsed time in [`Stats`](crate::Stats).
//!
//! With the default `clock` feature it is `web_time`'s, which on
//! `wasm32-unknown-unknown` calls the JavaScript host's `performance.now()`.
//! Without it the crate never reads a clock, for hosts that have none: a time
//! limit is then refused as an invalid option, and elapsed time stays zero.

#[cfg(feature = "clock")]
pub(crate) use web_time::Instant;

#[cfg(not(feature = "clock"))]
pub(crate) use none::Instant;

#[cfg(not(feature = "clock"))]
mod none {
    use std::time::Duration;

    /// A moment that is never read: every one is the same, none ends a
    /// limit, and none has passed time since another.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub(crate) struct Instant;

    impl Instant {
        pub(crate) fn now() -> Self {
            Instant
        }

        pub(crate) fn elapsed(&self) -> Duration {
            Duration::ZERO
        }

        pub(crate) fn checked_add(&self, _: Duration) -> Option<Self> {
            None
        }
    }
}

/// Refuses a time limit when there is no clock to keep it.
pub(crate) fn check_time_limit(
    field: &str,
    time_limit: Option<std::time::Duration>,
) -> Result<(), crate::Error> {
    if cfg!(feature = "clock") || time_limit.is_none() {
        return Ok(());
    }
    Err(crate::Error::InvalidOptions(format!(
        "invalid {field}: a time limit needs the `clock` feature"
    )))
}
