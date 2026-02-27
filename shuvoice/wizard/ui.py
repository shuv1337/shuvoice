"""Wizard UI helpers."""

from __future__ import annotations

from pathlib import Path

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, Gtk

from ..branding import logo_candidates

_LOGO_CANDIDATES = logo_candidates()


def find_logo() -> Path | None:
    for candidate in _LOGO_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def setup_css() -> None:
    css = Gtk.CssProvider()
    css.load_from_string(
        """window.wizard-window { background-color: rgba(15, 15, 20, 0.95); }
.wizard-page {
  padding: 48px 64px;
}
.wizard-title {
  color: white;
  font-size: 28px;
  font-weight: bold;
}
.wizard-subtitle {
  color: rgba(255, 255, 255, 0.7);
  font-size: 16px;
  margin-top: 4px;
  margin-bottom: 16px;
}
.wizard-desc {
  color: rgba(255, 255, 255, 0.55);
  font-size: 14px;
}
.wizard-radio label {
  color: white;
  font-size: 16px;
}
.wizard-radio-desc {
  color: rgba(255, 255, 255, 0.55);
  font-size: 13px;
  margin-left: 28px;
  margin-bottom: 8px;
}
.wizard-btn {
  padding: 8px 24px;
  font-size: 15px;
  border-radius: 8px;
  background-color: rgba(255, 255, 255, 0.12);
  color: white;
}
.wizard-btn:hover {
  background-color: rgba(255, 255, 255, 0.2);
}
.wizard-btn:focus {
  background-color: rgba(255, 255, 255, 0.25);
  outline: 2px solid white;
}
.wizard-btn-primary {
  background-color: rgba(60, 120, 220, 0.85);
}
.wizard-btn-primary:hover {
  background-color: rgba(60, 120, 220, 1.0);
}
.wizard-btn-primary:focus {
  background-color: rgba(60, 120, 220, 1.0);
  outline: 2px solid white;
}
.wizard-summary {
  color: rgba(255, 255, 255, 0.8);
  font-size: 15px;
  font-family: monospace;
  background-color: rgba(255, 255, 255, 0.06);
  border-radius: 8px;
  padding: 16px 20px;
  margin-top: 12px;
}
"""
    )
    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(),
        css,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
    )
