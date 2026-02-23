# PLAN: Issue #23 — Branding image optimization

Issue: https://github.com/shuv1337/shuvoice/issues/23  
Scope: `docs/assets/branding/*` plus any references in docs/runtime code  
Plan type: Documentation + asset optimization workflow (no feature behavior changes)

## Context

Issue #23 reports that branding assets under `docs/assets/branding/` are significantly larger than required for how they are rendered in repository docs.

The issue’s acceptance criteria require:
- right-sized branding assets for README/docs display,
- optimized exports (PNG/WebP as appropriate) without visible quality loss,
- correct light/dark rendering in `README.md` and `docs/BRANDING.md`, and
- at least **50% total size reduction** for `docs/assets/branding/`.

Important implementation reality discovered during repo inspection: these same branding PNGs are also loaded at runtime by GTK UI modules (`shuvoice/splash.py`, `shuvoice/wizard.py`), not just docs.

## Goals

1. Reduce total size of `docs/assets/branding/` from **2,761,739 bytes** to **<= 1,380,869 bytes** (minimum 50% reduction), with a preferred stretch target <= 1,200,000 bytes.
2. Keep visual quality acceptable at actual display sizes used by:
   - README logo (`width="760"`),
   - docs gallery previews, and
   - runtime splash/wizard logo rendering (`set_size_request(300)` / `set_size_request(320)`).
3. Preserve filename stability by default (same filenames/paths), minimizing downstream break risk.
4. Add a reproducible optimization/export workflow so future asset updates do not regress file size.

## Non-goals

- Redesigning logos, colors, typography, or branding layout.
- Changing splash/wizard UI behavior beyond asset file loading compatibility.
- Introducing a full asset pipeline framework or design-tool integration.
- Optimizing unrelated media outside `docs/assets/branding/`.

## Current-state findings

### 1) Exact asset inventory (current)

Baseline (`docs/assets/branding/`): **2,761,739 bytes (2.634 MiB)**

| Asset | Dimensions | Exact size | Share of branding dir |
|---|---:|---:|---:|
| `shuvoice-variant-dark-lockup.png` | 1536×1024 | 1,085,928 bytes | 39.3% |
| `shuvoice-variant-light-lockup.png` | 756×512 | 604,724 bytes | 21.9% |
| `shuvoice-variant-dark-badge.png` | 757×512 | 537,685 bytes | 19.5% |
| `shuvoice_variant_dark_lockup_alt.png` | 759×512 | 533,402 bytes | 19.3% |

### 2) Exact usage map (current references)

#### Documentation usage
- `README.md`
  - line 5: dark source uses `docs/assets/branding/shuvoice-variant-dark-lockup.png`
  - line 6: light source uses `docs/assets/branding/shuvoice-variant-light-lockup.png`
  - line 7: fallback `<img>` uses dark lockup at `width="760"`
- `docs/BRANDING.md`
  - lists dark badge, light lockup, dark lockup in asset list
  - includes gallery previews for those 3 files
  - includes copied README `<picture>` snippet with dark/light lockups

#### Runtime usage (non-doc consumers)
- `shuvoice/splash.py`
  - line 33 candidate: `shuvoice-variant-dark-lockup.png`
  - line 38 fallback candidate: `shuvoice-variant-dark-badge.png`
  - image displayed at `set_size_request(300, -1)`
- `shuvoice/wizard.py`
  - line 45 candidate: `shuvoice-variant-dark-lockup.png`
  - line 50 fallback candidate: `shuvoice-variant-dark-badge.png`
  - image displayed at `set_size_request(320, -1)`

#### Unreferenced asset
- `shuvoice_variant_dark_lockup_alt.png` has no in-repo references from `rg` (outside binary/build dirs).

### 3) Existing generation/optimization workflow status

- No dedicated script currently exists in `scripts/` for branding export/optimization.
- `docs/BRANDING.md` currently documents asset locations and README usage only.
- No documented size budgets, export settings, or optimization commands are present.

## Proposed approach

### Default strategy (recommended)

- Keep existing **PNG filenames and paths** stable.
- Optimize and (where needed) resize assets in-place to match actual display needs.
- Treat WebP as optional/secondary unless it clearly improves docs delivery without breaking runtime usage.

### Size targets (measurable)

Hard acceptance target (must pass):
- Total `docs/assets/branding/` <= **1,380,869 bytes**.

Per-asset working budgets (initial planning targets):
- `shuvoice-variant-dark-lockup.png`: <= 500,000 bytes (and reduce geometry from 1536 width).
- `shuvoice-variant-light-lockup.png`: <= 320,000 bytes.
- `shuvoice-variant-dark-badge.png`: <= 300,000 bytes.
- `shuvoice_variant_dark_lockup_alt.png`: <= 300,000 bytes if retained, or remove with rationale.

## Reproducible optimization workflow (tooling + commands)

> Planned to be codified in a script (e.g., `scripts/optimize-branding-assets.sh`) and referenced in `docs/BRANDING.md`.

### Tooling

- Required:
  - ImageMagick (`magick`, `identify`)
- Recommended:
  - `oxipng` for stronger PNG lossless optimization
  - `cwebp` for optional docs-only WebP variants

Example install (Arch):
```bash
sudo pacman -S imagemagick oxipng libwebp
```

### Baseline capture

```bash
find docs/assets/branding -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
file docs/assets/branding/*
```

### Safe working-copy flow

```bash
mkdir -p build/branding-backup build/branding-work build/branding-qa
cp docs/assets/branding/*.png build/branding-backup/
```

### Resize + optimize (deterministic sequence)

```bash
# Example: downscale oversized dark lockup first (keep aspect ratio)
magick docs/assets/branding/shuvoice-variant-dark-lockup.png \
  -filter Lanczos -resize 960x \
  -strip -define png:compression-level=9 -define png:compression-filter=5 \
  build/branding-work/shuvoice-variant-dark-lockup.png

# Copy other files into working dir (or resize if chosen)
cp docs/assets/branding/shuvoice-variant-light-lockup.png build/branding-work/
cp docs/assets/branding/shuvoice-variant-dark-badge.png build/branding-work/
cp docs/assets/branding/shuvoice_variant_dark_lockup_alt.png build/branding-work/

# Lossless optimize all PNGs
oxipng -o 4 --strip all build/branding-work/*.png
```

### Optional WebP evaluation (docs-only)

```bash
for f in build/branding-work/*.png; do
  cwebp -q 90 -m 6 "$f" -o "${f%.png}.webp"
done
```

### Quality QA artifacts for light/dark backgrounds

```bash
for f in build/branding-work/*.png; do
  b=$(basename "$f" .png)
  magick -size 1400x900 xc:'#ffffff' "$f" -gravity center -composite "build/branding-qa/${b}-on-light.png"
  magick -size 1400x900 xc:'#0d1117' "$f" -gravity center -composite "build/branding-qa/${b}-on-dark.png"
done
```

### Publish (only after validation)

```bash
cp build/branding-work/*.png docs/assets/branding/
```

## Task breakdown (ordered)

### Phase 1 — Baseline + decision record
- [ ] Record current baseline sizes/dimensions in PR notes (exact bytes and totals).
- [ ] Confirm intended fate of `shuvoice_variant_dark_lockup_alt.png` (retain + optimize vs remove as unused).
- [ ] Set final target dimensions for each asset (especially dark lockup) based on real display sizes.

### Phase 2 — Add reproducible workflow documentation
- [ ] Add a repeatable optimization workflow section to `docs/BRANDING.md` (tools, commands, size budgets).
- [ ] Document the default policy: preserve existing filenames/paths and PNG compatibility.
- [ ] Document optional WebP policy and when to use it.

### Phase 3 — Asset optimization execution
- [ ] Create backup and working directories under `build/`.
- [ ] Downscale oversized `shuvoice-variant-dark-lockup.png` to agreed target width.
- [ ] Run lossless PNG optimization across branding PNGs.
- [ ] Evaluate `shuvoice_variant_dark_lockup_alt.png` decision and apply consistently.
- [ ] Replace repo assets only after QA checks pass.

### Phase 4 — Reference integrity and migration handling
- [ ] Run reference scan to confirm no broken paths:
  - `README.md`
  - `docs/BRANDING.md`
  - `shuvoice/splash.py`
  - `shuvoice/wizard.py`
- [ ] If filenames/formats changed, update all references and fallback ordering.
- [ ] Ensure runtime still has a PNG path candidate for GTK compatibility.

### Phase 5 — Validation + acceptance proof
- [ ] Produce before/after size table with exact byte deltas and total reduction %.
- [ ] Verify issue acceptance target (>=50% total reduction) with reproducible command output.
- [ ] Capture visual QA screenshots/artifacts for light and dark backgrounds.
- [ ] Verify README + BRANDING rendering for both theme variants.

## Validation checklist

- [ ] **Size target**: total `docs/assets/branding/` <= 1,380,869 bytes.
- [ ] **Per-file report** generated with exact bytes and % reduction.
- [ ] **README correctness**: `<picture>` still resolves dark/light variants correctly.
- [ ] **BRANDING gallery correctness**: all listed images render with no broken links.
- [ ] **Light/dark visual QA**: transparent edges and contrast are clean on both `#ffffff` and `#0d1117` backgrounds.
- [ ] **Runtime safety**: splash/wizard logo loading still works with existing candidate paths.
- [ ] **Reference scan clean**: no stale branding filenames in tracked source/docs.

Suggested validation commands:

```bash
# size accounting
find docs/assets/branding -maxdepth 1 -type f -printf '%f\t%s\n' | sort
python - <<'PY'
from pathlib import Path
p = Path('docs/assets/branding')
sizes = sorted((f.name, f.stat().st_size) for f in p.iterdir() if f.is_file())
total = sum(s for _, s in sizes)
for n, s in sizes:
    print(f"{n}: {s}")
print(f"TOTAL: {total}")
PY

# references
rg -n "docs/assets/branding|shuvoice-variant|shuvoice_variant_dark_lockup_alt" README.md docs/BRANDING.md shuvoice/splash.py shuvoice/wizard.py
```

## Filename/format migration steps (only if stability policy is intentionally changed)

Default expectation: **do not rename files**.

If renaming or switching a consumer to WebP is necessary:
1. Keep legacy PNG files during migration window (do not remove immediately).
2. Update `README.md` and `docs/BRANDING.md` references in same commit.
3. Update `_LOGO_CANDIDATES` in:
   - `shuvoice/splash.py`
   - `shuvoice/wizard.py`
4. Ensure PNG fallback remains first-class for runtime GTK loading.
5. Run reference scan and manual runtime check before merging.
6. Document migration rationale and compatibility impact in PR description.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Over-compression artifacts (haloing/banding) | Brand quality regression | Use conservative settings first, inspect QA composites on light/dark backgrounds, iterate until clean. |
| Breaking runtime logo loading via format/path changes | Splash/wizard loses branding or fails to render | Preserve PNG filenames by default; if changing, update `_LOGO_CANDIDATES` and keep PNG fallback. |
| Unclear status of unused `shuvoice_variant_dark_lockup_alt.png` | Drift/noise in branding directory | Make explicit retain/remove decision in PR with rationale and usage scan evidence. |
| Future regressions from undocumented process | Asset sizes grow again | Add workflow + size budget guidance to `docs/BRANDING.md`. |
| Tool availability differences across dev machines | Non-reproducible results | Document required tools and pinned command sequence; keep fallback `magick`-only path noted. |

## Rollback strategy

1. Keep optimization changes isolated to a single commit touching branding assets/docs only.
2. Preserve pre-change originals in `build/branding-backup/` during implementation.
3. If regression is found post-merge:
   - immediate rollback via `git revert <branding-optimization-commit>`.
4. If only one asset regresses, restore selectively from git history:
   - `git checkout <known-good-ref> -- docs/assets/branding/<file>`.
5. Re-run validation checklist after rollback/partial restore.

## Definition of done

All of the following must be true before this issue can be closed:

1. **Size target met**: total `docs/assets/branding/` is <= 1,380,869 bytes (>= 50% reduction from 2,761,739 baseline).
2. **Per-asset size report**: a before/after table with exact byte counts and % reduction is included in the PR description.
3. **Visual quality preserved**: QA composites on `#ffffff` (light) and `#0d1117` (dark) backgrounds show no visible artifacts, haloing, or banding at display size.
4. **README rendering correct**: the `<picture>` block in `README.md` resolves dark/light variants correctly on GitHub (verified manually or via rendered preview).
5. **BRANDING.md rendering correct**: all gallery images in `docs/BRANDING.md` render with no broken links.
6. **Runtime logo loading intact**: `shuvoice/splash.py` and `shuvoice/wizard.py` `_LOGO_CANDIDATES` paths still resolve to valid PNG files.
7. **Reference scan clean**: `rg` over tracked source/docs shows no stale or broken branding file references.
8. **Optimization workflow documented**: `docs/BRANDING.md` includes a reproducible optimization section with tools, commands, and size budget guidance to prevent future regressions.
9. **`shuvoice_variant_dark_lockup_alt.png` disposition resolved**: an explicit retain-and-optimize or remove decision has been made with rationale documented in the PR.
10. **Single atomic commit**: all branding asset changes and reference updates are in one reviewable commit (or a small, logically ordered commit series).

## External tooling references

- ImageMagick: https://imagemagick.org/
- oxipng (lossless PNG optimizer): https://github.com/shssoichiro/oxipng
- WebP encoder (`cwebp`): https://developers.google.com/speed/webp/docs/cwebp
