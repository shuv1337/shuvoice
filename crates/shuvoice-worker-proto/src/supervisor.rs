//! Optional multi-attempt supervisor built on [`WorkerProcess`] + [`RestartState`].
//!
//! The supervisor never loops forever on its own: callers invoke
//! [`WorkerSupervisor::ensure_running`] / [`WorkerSupervisor::restart_after_failure`]
//! and honor [`RestartDecision::Wait`] / [`GiveUp`].
//!
//! Backoff is enforced internally via `next_retry_at` so repeated `ensure_running`
//! calls cannot bypass the policy delay. A death observed by `ensure_running` is
//! counted once; a subsequent `restart_after_failure` without an intervening
//! spawn does not double-count that same death. Each failed spawn attempt is
//! always counted.

use std::time::{Duration, Instant};

use crate::error::WorkerProcessError;
use crate::process::{WorkerExitStatus, WorkerProcess, WorkerSpawnConfig};
use crate::restart::{RestartDecision, RestartPolicy, RestartState};

/// Supervises a single worker slot with explicit restart decisions.
pub struct WorkerSupervisor {
    config: WorkerSpawnConfig,
    policy: RestartPolicy,
    state: RestartState,
    process: Option<WorkerProcess>,
    /// Instant the current process completed handshake.
    healthy_since: Option<Instant>,
    /// Clock origin for converting Instant → ms for the pure state machine.
    origin: Instant,
    /// Earliest time a new spawn may begin after a counted failure.
    next_retry_at: Option<Instant>,
    /// Set when a death/failure was already counted and no spawn has run since.
    /// Consumed by [`Self::restart_after_failure`] to avoid double-counting.
    failure_already_counted: bool,
}

impl WorkerSupervisor {
    #[must_use]
    pub fn new(config: WorkerSpawnConfig, policy: RestartPolicy) -> Self {
        Self {
            config,
            policy,
            state: RestartState::new(),
            process: None,
            healthy_since: None,
            origin: Instant::now(),
            next_retry_at: None,
            failure_already_counted: false,
        }
    }

    #[must_use]
    pub fn policy(&self) -> &RestartPolicy {
        &self.policy
    }

    #[must_use]
    pub fn restart_state(&self) -> &RestartState {
        &self.state
    }

    #[must_use]
    pub fn process(&self) -> Option<&WorkerProcess> {
        self.process.as_ref()
    }

    #[must_use]
    pub fn process_mut(&mut self) -> Option<&mut WorkerProcess> {
        self.process.as_mut()
    }

    /// When the next spawn attempt is allowed (`None` = immediately, subject to give-up).
    #[must_use]
    pub fn next_retry_at(&self) -> Option<Instant> {
        self.next_retry_at
    }

    /// Take ownership of the current process slot (e.g. for custom kill paths).
    pub fn take_process(&mut self) -> Option<WorkerProcess> {
        self.healthy_since = None;
        self.process.take()
    }

    /// Manually clear give-up / failure counters and backoff gate.
    pub fn reset_restart_state(&mut self) {
        self.state.reset();
        self.next_retry_at = None;
        self.failure_already_counted = false;
    }

    fn now_ms(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// If a backoff gate is active, return `RestartDeferred` with remaining delay.
    fn enforce_backoff_gate(&self) -> Result<(), WorkerProcessError> {
        if let Some(at) = self.next_retry_at {
            let now = Instant::now();
            if now < at {
                return Err(WorkerProcessError::RestartDeferred {
                    delay: at.saturating_duration_since(now),
                });
            }
        }
        if self.state.has_given_up() {
            return Err(WorkerProcessError::RestartExhausted {
                consecutive_failures: self.state.consecutive_failures,
            });
        }
        Ok(())
    }

    /// Count a failure and arm backoff / give-up.
    fn arm_failure(&mut self) -> Result<(), WorkerProcessError> {
        let decision = self.state.decide_after_failure(self.now_ms(), &self.policy);
        self.failure_already_counted = true;
        match decision {
            RestartDecision::GiveUp {
                consecutive_failures,
            } => {
                self.next_retry_at = None;
                Err(WorkerProcessError::RestartExhausted {
                    consecutive_failures,
                })
            }
            RestartDecision::Wait(delay) => {
                self.next_retry_at = Some(Instant::now() + delay);
                Err(WorkerProcessError::RestartDeferred { delay })
            }
            RestartDecision::RunNow => {
                self.next_retry_at = None;
                Ok(())
            }
        }
    }

    /// Ensure a live process exists. Spawns at most once per call.
    pub async fn ensure_running(&mut self) -> Result<&mut WorkerProcess, WorkerProcessError> {
        if let Some(proc) = self.process.as_mut() {
            match proc.try_status()? {
                None => {
                    if self.healthy_since.is_some() {
                        self.state.note_still_healthy(self.now_ms(), &self.policy);
                    }
                    Ok(self.process.as_mut().expect("just checked"))
                }
                Some(_status) => {
                    let dead = self.process.take().expect("just checked");
                    let exit = dead.kill().await;
                    self.healthy_since = None;
                    let crash = WorkerProcessError::Crashed {
                        exit_code: exit.code,
                        stderr_tail: exit.stderr_tail,
                    };
                    match self.arm_failure() {
                        // Prefer crash details; backoff is still armed when deferred.
                        Ok(()) | Err(WorkerProcessError::RestartDeferred { .. }) => Err(crash),
                        Err(WorkerProcessError::RestartExhausted {
                            consecutive_failures,
                        }) => Err(WorkerProcessError::RestartExhausted {
                            consecutive_failures,
                        }),
                        Err(other) => Err(other),
                    }
                }
            }
        } else {
            self.enforce_backoff_gate()?;
            self.spawn_fresh().await
        }
    }

    /// Record a failure (if not already counted for this cycle), apply backoff, spawn.
    pub async fn restart_after_failure(
        &mut self,
        exit: Option<WorkerExitStatus>,
    ) -> Result<&mut WorkerProcess, WorkerProcessError> {
        let _ = exit;
        if let Some(proc) = self.process.take() {
            let _ = proc.kill().await;
            self.healthy_since = None;
            // Caller killed a live process — always a new failure.
            self.arm_failure()?;
        } else if self.failure_already_counted {
            // Death/spawn-fail already counted (e.g. by ensure_running).
            self.failure_already_counted = false;
            self.healthy_since = None;
            self.enforce_backoff_gate()?;
        } else {
            self.healthy_since = None;
            self.arm_failure()?;
        }

        self.enforce_backoff_gate()?;
        self.spawn_fresh().await
    }

    /// Like [`restart_after_failure`] but sleeps when the policy asks to wait.
    pub async fn restart_after_failure_and_wait(
        &mut self,
        exit: Option<WorkerExitStatus>,
    ) -> Result<&mut WorkerProcess, WorkerProcessError> {
        let _ = exit;
        if let Some(proc) = self.process.take() {
            let _ = proc.kill().await;
            self.healthy_since = None;
            match self.arm_failure() {
                Ok(()) => {}
                Err(WorkerProcessError::RestartDeferred { delay }) => {
                    tokio::time::sleep(delay).await;
                }
                Err(e) => return Err(e),
            }
        } else if self.failure_already_counted {
            self.failure_already_counted = false;
            self.healthy_since = None;
            if let Some(at) = self.next_retry_at {
                let now = Instant::now();
                if now < at {
                    tokio::time::sleep(at - now).await;
                }
            }
            if self.state.has_given_up() {
                return Err(WorkerProcessError::RestartExhausted {
                    consecutive_failures: self.state.consecutive_failures,
                });
            }
        } else {
            self.healthy_since = None;
            match self.arm_failure() {
                Ok(()) => {}
                Err(WorkerProcessError::RestartDeferred { delay }) => {
                    tokio::time::sleep(delay).await;
                }
                Err(e) => return Err(e),
            }
        }

        self.spawn_fresh().await
    }

    /// Gracefully stop the current process if any.
    pub async fn shutdown(&mut self) -> Result<Option<WorkerExitStatus>, WorkerProcessError> {
        self.healthy_since = None;
        match self.process.take() {
            Some(proc) => Ok(Some(proc.shutdown().await?)),
            None => Ok(None),
        }
    }

    async fn spawn_fresh(&mut self) -> Result<&mut WorkerProcess, WorkerProcessError> {
        if let Some(at) = self.next_retry_at {
            if Instant::now() >= at {
                self.next_retry_at = None;
            } else {
                return Err(WorkerProcessError::RestartDeferred {
                    delay: at.saturating_duration_since(Instant::now()),
                });
            }
        }

        self.state.on_spawn_start(self.now_ms());
        match WorkerProcess::spawn(self.config.clone()).await {
            Ok(proc) => {
                self.state.on_healthy(self.now_ms());
                self.healthy_since = Some(Instant::now());
                self.next_retry_at = None;
                self.failure_already_counted = false;
                self.process = Some(proc);
                Ok(self.process.as_mut().expect("just set"))
            }
            Err(err) => {
                // Every failed spawn attempt counts toward the budget.
                match self.arm_failure() {
                    Ok(()) => Err(err),
                    Err(WorkerProcessError::RestartDeferred { delay }) => {
                        let _ = err;
                        Err(WorkerProcessError::RestartDeferred { delay })
                    }
                    Err(e) => Err(e),
                }
            }
        }
    }
}

/// Convenience: sleep helper for callers honoring [`RestartDecision::Wait`].
pub async fn honor_delay(delay: Duration) {
    if !delay.is_zero() {
        tokio::time::sleep(delay).await;
    }
}

impl std::fmt::Debug for WorkerSupervisor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerSupervisor")
            .field("policy", &self.policy)
            .field("state", &self.state)
            .field("has_process", &self.process.is_some())
            .field("healthy_since", &self.healthy_since)
            .field("next_retry_at", &self.next_retry_at)
            .field("failure_already_counted", &self.failure_already_counted)
            .finish_non_exhaustive()
    }
}
