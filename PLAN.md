# Native MPRIS pause/resume during recording

Status: implemented in the shared feature tree, with review corrections on
2026-10-12 PDT. Sections 1–7 retain the approved design; section 8 records actual
implementation choices and regression coverage. Tests use fakes and private
D-Bus only; no user playback, live configuration, or services were changed.

References: `reports/Automatic media pause during recording.md` and
`research_notes/Automatic media pause during recording/shuvoice-integration.md`
(plus `linux-media.md`). Root `AGENTS.md` applies; no nested instructions were
found in the relevant crate, settings, docs, or examples ancestor directories.

## 1. Scope and fixed shared contract

Add exactly one opt-in setting:

```toml
[audio]
pause_media_during_recording = false
```

- Rust field: `Config::pause_media_during_recording: bool`, default `false`.
- Settings ID: `audio.pause_media_during_recording`; Audio section, Bool,
  label **Pause media while recording**. Help/docs explain conservative automatic
  restoration after the microphone tail closes.
- Keep config version 1 and existing config migration/atomic patch semantics.
- Recording-only scope: pause supported initially playing MPRIS endpoints;
  resume only confirmed owned pauses. External media may resume before ShuVoice
  TTS begins. Preserve existing speech preroll unchanged.
- No shell/playerctl dependency, toggles, Stop calls, extensions, remote Spotify
  API, new control commands, or persistent restore ledger.
- Start remains usable if media control fails. Never count a media failure toward
  the ASR circuit breaker or convert it into dependency exit 78.
- Document best effort: browser endpoints are selected sessions, not all tabs;
  metadata cannot prove tab identity, buffered preroll can contain media, an
  unobservable manual Pause cannot be distinguished, and SIGKILL cannot restore.
  To suppress restoration within a hold, independently resume/stop/change the
  media (observed conflicts revoke entitlement); disabling the setting prevents
  subsequent acquisitions after the normal config reload/restart mechanism.

## 2. Architecture and interfaces

### Core: configuration and pure media values/predicates

New `crates/shuvoice-core/src/media.rs`, exported by `lib.rs`, owns plain transport
values and conservative predicates, with no Tokio or D-Bus types:

- `RecordingIntentId(u64)` distinct from `UtteranceGen`.
- `MediaEndpoint { bus_epoch, well_known_name, unique_owner, object_path }`.
  Canonical ledger key is `(bus_epoch, unique_owner, object_path)`; retain the
  original well-known name for fresh owner validation.
- `MediaSnapshot { playback, can_control, can_pause, can_play, fingerprint }`.
- `MediaFingerprint`: normalized optional track ID, URL, title, artist, album,
  length; omit artwork and transient position. Values are ephemeral and never
  logged. Empty/NoTrack/malformed identity cannot authorize restoration. Require
  at least a usable track ID or nonempty descriptive identity, compare the full
  selected fingerprint exactly, and treat constant browser IDs as heuristic only.
- Pure `eligible_to_pause` and `eligible_to_restore` predicates. Restore requires
  confirmed entitlement, unchanged owner/epoch/identity, fresh Paused status and
  `CanControl && CanPlay`, with no observed conflicting activity.
- Typed operation result distinguishes success, definitive rejection, and
  unknown outcome; raw D-Bus errors/metadata do not become diagnostic strings.
- Require a restoration-usable baseline fingerprint before Pause as well; skip
  identity-less endpoints instead of knowingly leaving them paused at release.

### App: transport seam and owned asynchronous coordinator

New `crates/shuvoice-app/src/media.rs` owns coordination. Add an object-safe async
`MediaTransport: Send + Sync` seam (using the existing async-trait dependency):

```rust
async fn subscribe(&self) -> Result<MediaEventReceiver, MediaError>;
async fn discover(&self) -> Result<Vec<MediaEndpoint>, MediaError>;
async fn snapshot(&self, endpoint: &MediaEndpoint) -> Result<MediaSnapshot, MediaError>;
async fn owner_matches(&self, endpoint: &MediaEndpoint) -> Result<bool, MediaError>;
async fn pause(&self, endpoint: &MediaEndpoint) -> MediaCallOutcome;
async fn play(&self, endpoint: &MediaEndpoint) -> MediaCallOutcome;
```

`MediaEventReceiver` reports owner loss/replacement, property changes or
invalidations, Seeked, and connection/subscription loss. Its reliable delivery or
explicit overflow/loss marker is part of the contract: loss invalidates affected
entitlements rather than silently missing user activity. Subscription installation
finishes before discovery. Callbacks match sender, object path, and interface.

`spawn_media_coordinator(transport)` returns `(MediaHandle, MediaCoordinatorJoin)`.
`MediaHandle` synchronously publishes the latest desired state via `watch`:
`Idle`, `Hold { intent }`, or terminal `Shutdown`. It exposes token-tagged
`Pending` / `Settled { intent, degraded }` readiness via a second watch channel.
No ignored bounded `try_send`, status-event observer, or bus await on the session
actor. All operation completions remain coordinator-owned independently of the
latest desired intent. One in-flight effect per endpoint; bounded concurrency
across endpoints. Use owned task tracking/JoinSet, not detached effect tasks.

Initial constants: 400 ms total acquisition readiness budget, 250 ms bounded
read-only calls, 1 s effect-call completion/confirmation ceiling, 2 s overall
coordinator shutdown budget. These are tunable implementation constants, not
user config or measured guarantees. Acquisition deadline means recording fails
open; it does not cancel a dispatched effect or erase its ledger entry. Stop
scheduling fresh pauses for an expired acquisition, while retaining dispatched
operations to settlement. Bound discovery/work by the whole acquisition deadline,
not a serial per-player multiplied timeout.

### IO: native asynchronous zbus adapter

New `crates/shuvoice-io/src/media.rs` implements `MprisClient` with the above core
DTOs. IO depends on core, not app. A small CLI `compose/media.rs` wrapper implements
the app transport trait by delegation and maps subscription delivery to app events;
this preserves the current crate dependency direction.

- Add workspace zbus dependency with Tokio support and default async-io runtime
  disabled; verify selected release features/MSRV against Rust 1.92 before locking.
- Add optional IO `mpris` feature and CLI `media-control = ["shuvoice-io/mpris"]`;
  include `media-control` in `desktop`. No GTK dependency. Feature-off builds use
  a settled/degraded no-op when configured on, without new exit-78 behavior.
- Connect lazily only when enabled and acquisition is needed. Production uses
  session bus; test constructor requires an explicit private-bus address.
- Enumerate owned `org.mpris.MediaPlayer2.*` names only; exclude playerctld's
  synthetic endpoint (including its instance suffix forms). Deduplicate aliases
  by unique owner/object. Do not follow new players appearing during the hold.
- Use explicit unique-owner destinations and no auto-activation. Read uncached
  properties; fresh-read invalidations. Explicit `Pause` and `Play` are the only
  mutation methods. Never infer success solely from method return.
- Disconnect increments bus epoch and invalidates all old entries. Reconnect may
  serve a future admission, never restore across an epoch or chase players during
  an existing recording.

## 3. Coordinator state machine and restoration ledger

1. On admitted intent, subscribe then snapshot eligible endpoints. Immediately
   before dispatch, recheck desired hold and relevant owner/state/identity. Pause
   only endpoints observed Playing with CanControl and CanPause.
2. Record operation ID, endpoint, baseline fingerprint, and originating intent
   before dispatch. A successful reply plus bounded fresh Paused confirmation,
   unchanged identity/owner, and no conflicting event creates entitlement.
   Signal handling must distinguish the expected Playing→Paused transition from
   subsequent conflicting Playing/Stopped/Seeked/media-change events. Reject
   ambiguous ordering rather than authorizing an unsafe restore.
3. Successful late Pause is still an owned effect: if latest desired state is
   Idle, reconcile to restore; if B is holding, transfer the entitlement to B.
   Inherited entries remain even though B's discovery sees Paused. Do not erase
   the initial entitlement or resume between A and B.
4. Unknown timeout/disconnection outcomes do not grant entitlement or permit
   retries/Play based only on a later Paused read. Report degraded control and
   abandon uncertain restoration; media may remain paused. If a definitive late
   reply is still being tracked within the effect ceiling, process it before
   classifying the operation unknown. Never claim timeout cancels remote effects.
5. On release, verify original name ownership and take a fresh snapshot. Restore
   once only if entitlement remains valid. Drain/reconcile pending invalidations
   and recheck the latest desired state immediately before Play dispatch. If a
   newer hold exists, retain rather than play. Clear entitlement after the play
   attempt, including unknown results; never blindly retry Play.
6. If Play was already dispatched when B arrives, serialize settlement before a
   fresh eligible re-pause for B. B readiness waits for that reconciliation or
   becomes degraded at its global deadline. Do not claim a token can unsend Play.
7. Observed independent Playing, Stopped, Seeked, metadata/owner changes, or loss
   of event continuity revoke entitlement. Never fight an independent resume by
   repeatedly pausing during the same intent. No metadata in logs; diagnostics
   contain stable failure class/counts/timing only.

## 4. Exact session lifecycle integration

Modify `session.rs`, `finalize.rs`, `runtime.rs`, and exports in app `lib.rs`:

- Add optional `SessionDeps::with_media(MediaHandle)`; absent is immediately
  settled and preserves current behavior. Session owns current intent, monotonically
  increasing counter, and pending reset/media readiness state.
- In `request_start_recording_with_case`, allocate intent only after StartGate
  admission (both Allow and AsrDisabledNeedsRecovery branches). Set new Hold
  before cleaning up superseded capture, so old cancellation cannot publish a
  transient Idle over a newer intent. All close/release helpers compare tokens.
- Change lifecycle purposes to `StartRecording { intent }` and
  `DisabledRecovery { intent }`. Store exact purpose/token on ActiveLifecycle and
  match Reset completions by identity, not just enum discriminant/start_pending.
  Disabled recovery chains normal start with the same token. ASR recovery
  generations can change without changing media intent.
- Successful reset stores `reset_ready_gen`; `try_complete_start` calls existing
  `complete_start_recording` only when matching media acquisition is settled or
  the 400 ms budget expires. Poll readiness in `poll_jobs`/pump paths as well as
   tick, so direct `start_recording` helpers also work. Unavailable/failed
   coordinator settles degraded; recording remains responsive.
- Preserve PCM received after reset readiness during any added media-only wait;
  append it once at successful start and discard it on cancellation. Existing
  idle/noise-floor draining must not discard speech during this new wait.
- Pending stop, `fail_start_recording`, disabled-recovery failure, breaker-open
  `disable_asr`, and detected ASR owner death close/invalidate the matching intent.
  Nonterminal chunk/reset recovery retains the hold.
- Cancellation helper `cancel_finalize_and_reset_injector` closes discarded
  capture and releases only the old token. TTS takeover uses the same stop/tail
  path; its timeout cancellation closes capture before releasing.
- `Session::shutdown` closes capture and publishes release. Session Drop also
  signals release for panic/abort; runtime owns the independent bounded async
  cleanup and does not rely on Drop to await anything.

### One authoritative utterance capture-closed boundary

Today key-up clears `recording`; chunk completion can defer finalize, and finalize
and actor maintain separate 120 ms grace deadlines. Replace that duplicated
decision rather than adding a stop timer:

- Initialize utterance state at successful start, and arm its recording edge
  immediately (factor the synchronous body of `begin_utterance` if useful).
  This preserves preroll and ensures stop-before-first-tick still finalizes.
- Keep accepting tail PCM while a chunk job delays finalization.
- `ActiveFinalize` stores its intent and `late_tx: Option<UnboundedSender<_>>`.
  Actor alone owns grace deadline. At expiry, stop routing samples to that
  utterance, take/drop the sender, then invoke idempotent
  `close_capture(intent, reason)`. Closing on cancellation also drops the sender
  and discards/aborts that utterance before release.
- Finalize consumes the late channel until EOF, draining all already accepted
  chunks; remove its independent grace timer and `grace` argument. Receiving
  buffered pre-close audio after release is safe: no new microphone samples can
  enter that closed utterance. Release does not wait for decode, network commit,
  synthesized tail noise, or transcript injection.
- Direct finalize/pump helpers must drive the same closure deadline, not hang
  awaiting a sender only tick can close. Tests use one authoritative fake-clock
  boundary; production tick remains 10 ms. Actor abort drops its sender.
- Runtime forces media shutdown only after actor exit/abort-and-await (therefore
  after microphone routing is impossible), not at the start of its join grace.

## 5. Composition and lifetime ownership

Keep existing `spawn_session_runtime` and `_with_transcripts` wrappers. Add
`spawn_session_runtime_with_services(..., RuntimeServices { transcripts, media })`
where media is an optional transport Arc; spawn the coordinator after ASR load
succeeds and retain its join owner in `SessionRuntime`. CLI constructs the lazy
native wrapper only for enabled config + compiled feature.

Existing `compose/mod.rs` startup-abort and normal cleanup already call runtime
shutdown after audio ingress is stopped. Extend runtime shutdown: actor normal
exit or forced abort-and-await, then terminal media shutdown and bounded join,
then remaining teardown (ASR/dispatcher can be joined concurrently where safe).
The terminal state cannot be overwritten by a surviving stale handle. Close and
join all coordinator-owned tasks at budget exhaustion and classify unresolved
remote effects unknown. No guarantee is made for work executed remotely after
timeouts or process death. Include command-channel closure and actor panic paths.

## 6. Targeted verification (future implementation)

Use fake transports/clocks and private fake MPRIS services only. Never connect a
test mutation to the user's session bus or invoke real playerctl Pause/Play.

1. **Config/settings:** absent/default false, true/false TOML round trip, legacy
   migration default, non-boolean rejection, effective config field, settings
   schema Bool/Audio placement and patch persistence preserving unrelated keys.
2. **Coordinator:** multiple players/partial failure; initially paused/stopped or
   incapable players untouched; Pause reply without Paused confirmation;
   owner replacement/alias dedup; metadata change/empty identity; Seeked and
   independent resume; invalidated properties/event loss; no new-player chasing;
   unknown timeout never followed by blind Play; disconnect epoch invalidation.
3. **Race tests:** A-stop during discovery/reset/pause; queued A reset completion
   after B admission; late successful pause with Idle vs B; queued release
   superseded by B; already-dispatched Play when B arrives; stalled transport
   preserves Stop/status responsiveness and bounded readiness/shutdown.
4. **Capture/lifecycle:** short tap before first tick; slow chunk extends stop
   edge before grace starts; burst-lossless late channel; release exactly after
   closure, before slow decode/injection; no post-release PCM enters old
   utterance; nonterminal recovery retains hold; reset failure/terminal ASR death,
   cancellation/supersession/TTS timeout/shutdown close it. Preserve existing
   `ship_blockers.rs` late-audio, slow-reset, and forced-shutdown regressions.
5. **Private-bus IO:** explicit address fixture, fake owned services implementing
   properties and Pause/Play counters, no-op Pause, delayed replies, owner restart,
   spoofed signals, subscription-before-snapshot, invalidations, reconnect,
   no auto-activation. Own/reap private bus process; skip with an explicit reason
   locally if dbus-daemon is unavailable and require it in the relevant CI job.
6. **Cleanup:** startup abort, actor panic/forced abort, shutdown with pending
   Pause/Play and bus loss; assert task joins and total bounded cleanup, not an
   impossible assertion that an unknown remote effect was undone.

After implementation run `cargo fmt --check`, core/app tests, IO tests with
`--no-default-features --features mpris`, CLI default and no-default-feature
checks/tests, and default CLI clippy with `-D warnings`. Report dependency/build
blockers honestly; do not install packages or mutate services to unblock checks.
Real Spotify/YouTube compatibility and latency acceptance require separately
authorized playback testing and are not certified by private-bus success.

## 7. Independent implementation subdivisions

**A — Config/settings/docs:** owns `shuvoice-core/src/config/{model,defaults}.rs`,
config tests, `shuvoice-core/src/settings.rs`, applicable CLI config/settings
contract tests, `examples/config.toml`, `docs/CONFIGURATION.md`, and root
`AGENTS.md`. Implement the exact field/ID in section 1. Update desktop feature
documentation to include `media-control` and best-effort behavior coordinated
with B. No TypeScript config logic or live config edits.

**B — Runtime/native transport:** owns core `media.rs` + `lib.rs` export, app
media/session/finalize/runtime/traits/export changes and lifecycle tests, IO
adapter/export/tests, CLI `compose/media.rs` and composition callsite, Cargo
manifests/lockfile. Consume **`config.pause_media_during_recording`** directly;
do not invent a second setting or duplicate config patches. Implement in order:
pure policy + fake coordinator → capture closure/intent integration → native
adapter/composition → adversarial/private-bus checks. B can work independently
but final compilation depends on A landing the shared config field.

Lead integrates both, checks the complete matrix, updates this plan's status,
and handles review/notifications. This planning change creates only `PLAN.md`.

## 8. Implementation and review corrections

- Readiness is tracked separately from the lifetime of held pause entitlements.
  Empty discovery, unavailable transport, ineligible players, and completed
  confirmations settle early; the 400 ms deadline remains an admission-time cap.
  A worker restoring an already-dispatched Play remains pending for a successor.
- Each endpoint task retains its originating intent and returns a replay reason.
  Undispatched failed/expired predecessor reads may be reacquired by a newer
  intent within that newer admission budget. Dispatched unknown effects and
  observed conflicts never authorize same-intent retries.
- Successful Pause replies are followed by bounded fresh confirmation polling
  (10 ms interval) within the single 1 s effect ceiling. A no-op response does
  not acquire an entitlement. Conflicting signals revoke it during confirmation.
  Confirmation is phase-aware: unchanged Playing, including fresh reads caused
  by benign invalidations, is permitted before the first Paused observation.
  Subsequent Playing revokes ownership. A predecessor-valued read that raced a
  newer target signal is re-read rather than misclassified as a reversal.
  A target-valued snapshot marks the target observed immediately on return,
  before owner validation or any further await/drain. Predecessor signals queued
  alongside that target snapshot are rejected conservatively when ordering cannot
  prove they preceded the snapshot; manual resume/re-pause during owner validation
  therefore cannot recreate restoration entitlement.
- PropertiesChanged preserves relevant playback, fingerprint, capability, and
  invalidation values. Batched Paused plus unchanged identity/capabilities or
  unrelated properties is accepted. Relevant invalidations trigger bounded fresh
  reads and signal re-drain; changed identity, conflicting playback, continuity
  loss, and persistent invalidation churn still revoke restoration. The native
  adapter fences its incoming signal stream after fresh property reads.
- Eight permits bound concurrent endpoint operations; up to 128 endpoint tasks
  may retain ledgers. Local restore-capacity waits retain confirmed entitlement
  rather than applying a remote-call timeout to semaphore contention. Remote
  calls remain bounded after obtaining capacity; overall shutdown remains 2 s.
- Play uses the same bounded transition confirmation after a successful reply.
  An endpoint remains pending while it still reports Paused; a newer hold cannot
  settle or launch its re-pause until Playing is freshly confirmed. Unknown,
  rejected, timed-out, or unconfirmed Play is not retried and does not authorize
  a blind re-pause. Transitions beyond the confirmation/admission deadlines can
  still escape best-effort protection; no remote cancellation is claimed.
- ASR death uses one idempotent teardown path shared by Start admission and tick,
  closing capture, lifecycle/chunk/finalize work, and media before publishing the
  dead-owner flag. Post-reset PCM is retained during media-only readiness waits.
- Capture grace has one actor-owned Tokio deadline, also driven by direct pump
  helpers. Finalize reads accepted late PCM to EOF; decoding/injection does not
  hold media. Paused-time tests drive the same deadline.
- Focused regressions cover early readiness/short utterances, 20 ms delayed
  player transitions, batched properties and benign invalidations, superseded
  undispatched reads, nine-player slow restoration contention, Start-before-tick
  ASR death, runtime shutdown during Pause, and rejection of post-close PCM.
  The private-bus fixture verifies preserved batched/invalidation data and signal
  fencing alongside alias, owner, epoch, and spoofed-sender checks. Real player
  compatibility and the broader proposed private-bus fault matrix remain separate
  acceptance work.

Review-correction validation (2026-10-12 PDT):

- `cargo test -p shuvoice-app -p shuvoice-core -p shuvoice-io --no-default-features --features shuvoice-io/mpris`: 359 passed, including 15 media integration regressions and the private-bus fixture; no failures/skips.
- CLI `cli_contract`, `compose_harness`, and `ipc_e2e`: 22 passed with default features; 21 passed with `--no-default-features`.
- `cargo check -p shuvoice-cli` and its `--no-default-features` variant passed.
- CLI clippy with `-D warnings`, default and feature-off, passed.
- `cargo fmt --all --check` and `git diff --check` passed.

Second ordering-review validation: the same combined app/core/IO command passed
362 tests (18 media integration regressions), with zero failures or skips. New
cases combine delayed Pause with benign metadata invalidation, verify that an
observed Paused→Playing reversal still revokes entitlement, delay Play's actual
transition independently of its reply while a successor waits, and ensure
unconfirmed/unknown Play never triggers blind retries. Default and feature-off
CLI clippy with `-D warnings`, formatting, and diff checks also passed.

Final observation-order correction: the combined app/core/IO suite passed 363
tests, including 19 media regressions. The added regression returns Paused from
the confirmation snapshot, blocks owner validation, then delivers manual
Playing→Paused signals and asserts no restoration. Both CLI clippy feature
variants, formatting, and diff checks passed again.
