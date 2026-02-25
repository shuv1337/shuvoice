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
        "window.wizard-window { background-color: rgba(15, 15, 20, 0.95); }\n"
        ".wizard-page {\n"
        "  padding: 48px 64px;\n"
        "}\n"
        ".wizard-title {\n"
        "  color: white;\n"
        "  font-size: 28px;\n"
        "  font-weight: bold;\n"
        "}\n"
        ".wizard-subtitle {\n"
        "  color: rgba(255, 255, 255, 0.7);\n"
        "  font-size: 16px;\n"
        "  margin-top: 4px;\n"
        "  margin-bottom: 16px;\n"
        "}\n"
        ".wizard-desc {\n"
        "  color: rgba(255, 255, 255, 0.55);\n"
        "  font-size: 14px;\n"
        "}\n"
        ".wizard-radio label {\n"
        "  color: white;\n"
        "  font-size: 16px;\n"
        "}\n"
        ".wizard-radio-desc {\n"
        "  color: rgba(255, 255, 255, 0.55);\n"
        "  font-size: 13px;\n"
        "  margin-left: 28px;\n"
        "  margin-bottom: 8px;\n"
        "}\n"
        ".wizard-btn {\n"
        "  padding: 8px 24px;\n"
        "  font-size: 15px;\n"
        "  border-radius: 8px;\n"
        "  background-color: rgba(255, 255, 255, 0.12);\n"
        "  color: white;\n"
        "}\n"
        ".wizard-btn:hover {\n"
        "  background-color: rgba(255, 255, 255, 0.2);\n"
        "}\n"
        ".wizard-btn-primary {\n"
        "  background-color: rgba(60, 120, 220, 0.85);\n"
        "}\n"
        ".wizard-btn-primary:hover {\n"
        "  background-color: rgba(60, 120, 220, 1.0);\n"
        "}\n"
        ".wizard-summary {\n"
        "  color: rgba(255, 255, 255, 0.8);\n"
        "  font-size: 15px;\n"
        "  font-family: monospace;\n"
        "  background-color: rgba(255, 255, 255, 0.06);\n"
        "  border-radius: 8px;\n"
        "  padding: 16px 20px;\n"
        "  margin-top: 12px;\n"
        "}\n"
    )
    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(),
        css,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
    )
