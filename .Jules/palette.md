## PALETTE'S JOURNAL

## 2025-02-17 - [GTK4 Overlay Accessibility]
**Learning:** Custom GTK4 layer-shell overlays are often invisible to screen readers unless explicitly updated. Static labels/icons are insufficient for dynamic states like "Listening" vs "Processing".
**Action:** Always expose state changes via `update_property([Gtk.AccessibleProperty.LABEL], [status_text])` on the primary interactive or status element.
