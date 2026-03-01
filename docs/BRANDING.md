# ShuVoice brand assets

ShuVoice logo assets used by the repository live in:

- `docs/assets/branding/shuvoice-variant-dark-badge.png`
- `docs/assets/branding/shuvoice-variant-light-lockup.png`
- `docs/assets/branding/shuvoice-variant-dark-lockup.png`

## Preview gallery

### Dark badge

![ShuVoice dark badge](assets/branding/shuvoice-variant-dark-badge.png)

### Light lockup

![ShuVoice light lockup](assets/branding/shuvoice-variant-light-lockup.png)

### Dark lockup

![ShuVoice dark lockup](assets/branding/shuvoice-variant-dark-lockup.png)

## README usage

The README uses a `picture` block to select a light/dark logo depending on viewer theme.

```html
<p align="center">
  <picture>
    <source
      media="(prefers-color-scheme: dark)"
      srcset="./docs/assets/branding/shuvoice-variant-dark-lockup.png"
    />
    <source
      media="(prefers-color-scheme: light)"
      srcset="./docs/assets/branding/shuvoice-variant-light-lockup.png"
    />
    <img
      src="./docs/assets/branding/shuvoice-variant-dark-lockup.png"
      alt="ShuVoice logo"
      width="760"
    />
  </picture>
</p>
```

## Asset optimization guide

To prevent size regressions when updating branding assets, follow this workflow.

### Requirements

```bash
sudo pacman -S imagemagick oxipng   # Arch Linux
```

### Target dimensions

| Asset                              | Max width | Rationale                                               |
| ---------------------------------- | --------: | ------------------------------------------------------- |
| `shuvoice-variant-dark-lockup.png` |     960px | README displays at 760px; 960 provides retina headroom  |
| All others                         |     640px | Runtime splash/wizard display at 320px; 640 provides 2× |

### Optimization workflow

```bash
# 1. Resize if source exceeds target dimensions
magick input.png -filter Lanczos -resize 960x \
  -strip -define png:compression-level=9 -define png:compression-filter=5 \
  output.png

# 2. Lossless PNG optimization
oxipng -o 6 --strip all docs/assets/branding/*.png

# 3. Verify sizes
du -b docs/assets/branding/*.png
```

### Size budget

Total `docs/assets/branding/` should stay under **1.4 MiB**.
Current total: ~916 KB (as of Feb 2026).
