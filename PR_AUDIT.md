# ShuVoice Open PR Audit & Merge Plan

_Date: 2026-03-02_

> Note: repo default branch is `master` (not `main`). All branch-age/conflict checks below are against `origin/master`.

## 1) Executive summary

- Open PRs audited: **14** (`#33`–`#46`)
- Immediately mergeable + green CI: **2** (`#33`, `#35`)
- Mergeable but failing lint/format checks: **3** (`#34`, `#40`, `#42`)
- Conflicting with `master`: **9**
- Strong duplicate clusters:
  - **Control socket DoS fix:** `#34`, `#36`, `#39`, `#44`
  - **Overlay accessibility fix:** `#35`, `#45` (+ overlap in `#37`)
  - **Wizard button focus styling:** `#40`, `#42`

Mainline has moved quickly (`git log --oneline -30` shows substantial config/wizard/sherpa churn), which explains the high conflict rate and noisy unrelated diffs in many PRs.

---

## 2) Summary table (all open PRs)

| PR | Title (short) | Merge/check state | Behind `master` | Category | Recommendation |
|---:|---|---|---:|---|---|
| 33 | Avoid redundant RMS calculation in ASR loop | CLEAN, checks pass | 21 | ✅ Merge | Merge as-is |
| 34 | DoS fix in control socket | MERGEABLE, lint fail | 21 | 🔧 Merge with fixes | Keep, fix formatting + tighten tests |
| 35 | Overlay caption accessibility | CLEAN, checks pass | 21 | ✅ Merge | Merge as-is |
| 36 | DoS fix in control socket | CONFLICTING | 12 | ❌ Close | Duplicate of #34/#39/#44 |
| 37 | Overlay + wizard accessibility relations | CONFLICTING | 11 | 🤔 Needs discussion | Split wizard-only a11y into fresh PR |
| 38 | Transcript overlap optimization (`rsplit`) | CONFLICTING | 11 | 🔧 Merge with fixes | Cherry-pick focused hunk only |
| 39 | DoS fix + security socket test | CONFLICTING, lint fail | 4 | ❌ Close | Duplicate; salvage test idea only |
| 40 | Wizard focus styles (`:focus`) | MERGEABLE, lint fail | 4 | ❌ Close | Superseded by #42; includes risky import behavior change |
| 41 | Utterance buffer optimization | CONFLICTING | 3 | ❌ Close | Logic bug + deletes test file |
| 42 | Wizard focus-visible styles | MERGEABLE, lint fail | 2 | 🔧 Merge with fixes | Keep; remove noise/artifacts and reformat |
| 43 | Moonshine RMS optimization (`np.dot`) | CONFLICTING | 2 | 🔧 Merge with fixes | Keep; clean/cherry-pick only functional hunk |
| 44 | DoS fix in control socket | CONFLICTING | 2 | ❌ Close | Duplicate of security cluster |
| 45 | Overlay accessibility (STATUS + label updates) | CONFLICTING | 2 | ❌ Close | Duplicate of #35 with extra churn |
| 46 | Backspace arg generation optimization | CONFLICTING | 2 | ❌ Close | Net no-op for `typer.py` (optimization reverted) |

---

## 3) Cross-PR findings

## Duplicates

### A) Control socket DoS fix duplicates
- PRs: **#34, #36, #39, #44**
- All attempt to set timeout on accepted `conn` sockets.
- Best base to keep: **#34** (smallest mergeable branch; needs formatting fix).
- `#39` has useful added regression test (`tests/test_security_socket.py`) but also extra churn and conflicting changes.

### B) Overlay accessibility duplicates
- PRs: **#35, #45** (+ overlapping part in `#37`)
- `#35` is the cleanest implementation (`AccessibleRole.STATUS` + `update_property` + test file) and already green/mergeable.
- `#45` is functionally duplicative and noisy/conflicting.

### C) Wizard focus style duplicates
- PRs: **#40, #42**
- `#42` is the better UX direction (`:focus-visible`), but needs cleanup.
- `#40` adds extra behavior change around optional `Gtk4LayerShell` import that should not ride with CSS accessibility.

## Conflicts / hotspot files

Most conflicts are from recurring, unrelated formatting churn in shared files (not feature logic):
- `shuvoice/asr_sherpa.py`, `shuvoice/config.py`, `shuvoice/wizard/__init__.py`, `shuvoice/wizard/actions.py`, `shuvoice/wizard_state.py`
- `tests/test_packaging_service_unit.py`, `tests/test_asr.py`, `tests/test_typer.py`, etc.

Recommendation: for conflicted “good” PRs, **do not merge branches as-is**; instead cherry-pick only the intended functional hunks into fresh PRs.

## Superseded / stale / outdated

- Several older branches are **21 commits behind** (`#33`, `#34`, `#35`), though still mergeable.
- `#46` is effectively superseded by itself: the final branch **does not contain** the advertised `typer.py` optimization.

---

## 4) Detailed notes per PR

### PR #33 — ⚡ Bolt: Avoid redundant RMS calculation in ASR loop
- Core change is valid: computes `chunk_rms` once in `transcribe_native_chunk()` and reuses it.
- Low risk perf improvement in hot path.
- Category: ✅ **Merge**.

### PR #34 — 🛡️ Sentinel: [HIGH] Fix DoS vulnerability in control socket
- Valuable security fix: `conn.settimeout(1.0)` + explicit `socket.timeout` handling.
- CI failure is formatting-only (`tests/test_packaging_service_unit.py`).
- Category: 🔧 **Merge with fixes** (format + ideally add dedicated regression test).

### PR #35 — 🎨 Palette: Improve accessibility of caption overlay
- Good a11y fix: STATUS role + accessible label updates on text change.
- Includes focused test coverage (`tests/test_overlay_accessibility.py`).
- Category: ✅ **Merge**.

### PR #36 — 🛡️ Sentinel: Fix DoS vulnerability in control socket
- Duplicate security fix; less complete than #34/#39.
- Conflicting and offers no unique value.
- Category: ❌ **Close**.

### PR #37 — 🎨 Palette: Improve accessibility for overlay and wizard
- Contains overlay fix (duplicate of #35) and unique wizard `DESCRIBED_BY` relations.
- Mixed scope, conflicting branch, no dedicated wizard a11y tests.
- Category: 🤔 **Needs discussion** (split wizard-only change into fresh PR).

### PR #38 — ⚡ Bolt: Optimize transcript overlap stitching for long sessions
- Functional hunk in `transcript.py` is good (`split()` -> bounded `rsplit()`).
- Performance improvement is credible; logic appears sound.
- Branch is noisy/conflicting due unrelated formatting.
- Category: 🔧 **Merge with fixes** (clean cherry-pick).

### PR #39 — feat(security): enforce timeout on control socket connections
- Security intent overlaps with #34/#44.
- Adds `tests/test_security_socket.py` (useful idea), but branch also removes module docstring and has high churn.
- CI red (format) and conflicting.
- Category: ❌ **Close** (salvage test concept into kept security PR).

### PR #40 — feat(wizard): Add focus styles to buttons
- Duplicate space with #42.
- Uses `:focus` and also changes wizard import behavior around `Gtk4LayerShell` (undesired coupling/risk).
- Category: ❌ **Close**.

### PR #41 — ⚡ Bolt: Optimize utterance buffer consumption
- Has real bug: fast path returns `has_more=True` unconditionally.
- Deletes `tests/test_utterance_state.py` instead of updating coverage.
- Category: ❌ **Close**.

### PR #42 — 🎨 Palette: Add focus-visible states to wizard buttons
- Good accessibility direction (`:focus-visible`).
- Needs cleanup: remove `.Jules/palette.md` artifact and unrelated test churn; run formatter.
- Category: 🔧 **Merge with fixes**.

### PR #43 — ⚡ Bolt: Optimize RMS calculation using np.dot
- Useful micro-optimization in `asr_moonshine.py` (`np.dot(buf, buf)/buf.size`).
- Branch includes lots of unrelated formatting/conflicting churn.
- Category: 🔧 **Merge with fixes** (focused cherry-pick only).

### PR #44 — 🛡️ Sentinel: [MEDIUM] Fix control socket DoS vulnerability
- Duplicate of security cluster.
- Conflicting and no differentiating value over a cleaned #34 path.
- Category: ❌ **Close**.

### PR #45 — 🎨 Palette: Improve accessibility of CaptionOverlay...
- Duplicates #35 core overlay changes.
- Conflicting and much noisier.
- Category: ❌ **Close**.

### PR #46 — ⚡ Bolt: optimize wtype backspace arguments generation
- Important finding: final PR diff does **not** include `shuvoice/typer.py` optimization.
- Commit `d46d5a6` adds optimization, but later commit `a3a1793` reverts it while keeping same message.
- Net result: no shipped feature + noisy unrelated changes.
- Category: ❌ **Close**.

---

## 5) Recommended merge order

1. **#33** (clean perf win, low risk)
2. **#35** (clean accessibility win)
3. **#34 (fixed)**: reformat + optionally pull in targeted regression test pattern from #39
4. **#42 (fixed)**: keep only CSS `:focus-visible` change, remove artifact/noise
5. **#38 (fixed)**: cherry-pick transcript optimization only
6. **#43 (fixed)**: cherry-pick moonshine RMS optimization only
7. **#37 (discussion)**: if approved, re-open as wizard-only `DESCRIBED_BY` PR with tests

---

## 6) PRs recommended to close now

- **Security duplicates:** `#36`, `#39`, `#44`
- **Overlay duplicate:** `#45`
- **Wizard focus duplicate/risky variant:** `#40`
- **Buggy optimization:** `#41`
- **No-op/misleading branch:** `#46`

---

## 7) Risks / concerns

- AI-generated branches include heavy unrelated formatting/test churn; direct merges will increase conflict and reduce review clarity.
- Multiple PRs bundle CI fixes unrelated to the stated feature, causing false conflict surfaces.
- Security fix should land quickly, but with one canonical PR only, to avoid repeated rebases and inconsistent timeout semantics.
- For remaining perf/a11y work, prefer **new clean follow-up PRs** over rebasing noisy historical branches.
