"""First-run welcome wizard for ShuVoice.

Presents a guided setup dialog on the first launch, allowing the user
to select an ASR backend and review the default IPC push-to-talk flow.
Writes ``config.toml`` and a ``.wizard-done`` marker so it only runs once.

Uses the Hyprland layer-shell system for visual consistency with the
rest of ShuVoice.

IMPORTANT: ctypes.CDLL('libgtk4-layer-shell.so') must be called
before this module is imported. See __main__.py.

Pure-logic helpers live in ``wizard_state.py`` so they can be tested
without a GTK runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path

import gi

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("Gtk4LayerShell", "1.0")
from gi.repository import Gdk, Gtk
from gi.repository import Gtk4LayerShell as LayerShell

from .wizard_state import (
    ASR_BACKENDS,
    KEYBIND_PRESETS,
    format_hyprland_bind,
    format_summary,
    needs_wizard,
    write_config,
    write_marker,
)

log = logging.getLogger(__name__)

_LOGO_CANDIDATES = [
    Path(__file__).resolve().parent.parent
    / "docs"
    / "assets"
    / "branding"
    / "shuvoice-variant-dark-lockup.png",
    Path(__file__).resolve().parent.parent
    / "docs"
    / "assets"
    / "branding"
    / "shuvoice-variant-dark-badge.png",
]

# Re-export for __main__.py
__all__ = ["WelcomeWizard", "needs_wizard"]


def _find_logo() -> Path | None:
    for candidate in _LOGO_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


class WelcomeWizard(Gtk.Application):
    """First-run guided setup wizard."""

    def __init__(self):
        super().__init__(application_id="io.github.shuv1337.shuvoice.wizard")
        self.completed = False
        self._asr_backend = "sherpa"
        self._keybind = "f9"
        self._win: Gtk.Window | None = None
        self._stack: Gtk.Stack | None = None

    # -- GTK lifecycle ---------------------------------------------------------

    def do_activate(self):
        win = Gtk.Window(application=self)
        self._win = win

        if LayerShell.is_supported():
            LayerShell.init_for_window(win)
            LayerShell.set_layer(win, LayerShell.Layer.TOP)
            # Avoid global keyboard grabs during setup; keeps input recoverable
            # even if window teardown is delayed by compositor timing.
            LayerShell.set_keyboard_mode(win, LayerShell.KeyboardMode.ON_DEMAND)
            LayerShell.set_exclusive_zone(win, -1)
            LayerShell.set_namespace(win, "shuvoice-wizard")

        self._setup_css()

        self._stack = Gtk.Stack()
        self._stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        self._stack.set_transition_duration(200)

        self._stack.add_named(self._build_welcome_page(), "welcome")
        self._stack.add_named(self._build_asr_page(), "asr")
        self._stack.add_named(self._build_keybind_page(), "keybind")
        self._stack.add_named(self._build_done_page(), "done")

        win.set_child(self._stack)
        win.add_css_class("wizard-window")
        win.present()

    def _setup_css(self):
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

    # -- Page builders ---------------------------------------------------------

    def _build_welcome_page(self) -> Gtk.Widget:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        page.add_css_class("wizard-page")
        page.set_halign(Gtk.Align.CENTER)
        page.set_valign(Gtk.Align.CENTER)
        page.set_spacing(4)

        logo_path = _find_logo()
        if logo_path:
            try:
                logo = Gtk.Picture.new_for_filename(str(logo_path))
                logo.set_can_shrink(True)
                logo.set_size_request(320, -1)
                logo.set_halign(Gtk.Align.CENTER)
                logo.set_margin_bottom(12)
                page.append(logo)
            except Exception:
                self._add_text_title(page, "ShuVoice")
        else:
            self._add_text_title(page, "ShuVoice")

        sub = Gtk.Label(label="Streaming speech-to-text for Hyprland")
        sub.add_css_class("wizard-subtitle")
        sub.set_halign(Gtk.Align.CENTER)
        page.append(sub)

        desc = Gtk.Label(
            label="Let\u2019s set up a few things before you start.\nThis will only take a moment."
        )
        desc.add_css_class("wizard-desc")
        desc.set_halign(Gtk.Align.CENTER)
        desc.set_justify(Gtk.Justification.CENTER)
        desc.set_margin_bottom(24)
        page.append(desc)

        btn = self._make_button("Get Started", primary=True)
        btn.connect("clicked", lambda _: self._stack.set_visible_child_name("asr"))
        btn.set_halign(Gtk.Align.CENTER)
        page.append(btn)

        return page

    def _build_asr_page(self) -> Gtk.Widget:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        page.add_css_class("wizard-page")
        page.set_halign(Gtk.Align.CENTER)
        page.set_valign(Gtk.Align.CENTER)
        page.set_spacing(4)

        self._add_text_title(page, "Speech Recognition Engine")

        sub = Gtk.Label(label="Choose the ASR backend that fits your hardware.")
        sub.add_css_class("wizard-subtitle")
        page.append(sub)

        group: Gtk.CheckButton | None = None
        self._asr_radios: dict[str, Gtk.CheckButton] = {}

        for backend_id, label, description in ASR_BACKENDS:
            radio = Gtk.CheckButton(label=label)
            radio.add_css_class("wizard-radio")
            if group is None:
                group = radio
            else:
                radio.set_group(group)

            if backend_id == "sherpa":
                radio.set_active(True)

            radio.connect("toggled", self._on_asr_toggled, backend_id)
            page.append(radio)
            self._asr_radios[backend_id] = radio

            desc_label = Gtk.Label(label=description)
            desc_label.add_css_class("wizard-radio-desc")
            desc_label.set_halign(Gtk.Align.START)
            page.append(desc_label)

        nav = self._make_nav_row(
            back_page="welcome",
            next_page="keybind",
        )
        nav.set_margin_top(20)
        page.append(nav)

        return page

    def _build_keybind_page(self) -> Gtk.Widget:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        page.add_css_class("wizard-page")
        page.set_halign(Gtk.Align.CENTER)
        page.set_valign(Gtk.Align.CENTER)
        page.set_spacing(4)

        self._add_text_title(page, "Push-to-Talk Keybind")

        sub = Gtk.Label(
            label="ShuVoice uses Hyprland bind/bindr for push-to-talk.\n"
            "Hold the key to record, release to stop."
        )
        sub.add_css_class("wizard-subtitle")
        sub.set_justify(Gtk.Justification.CENTER)
        page.append(sub)

        group: Gtk.CheckButton | None = None
        self._keybind_radios: dict[str, Gtk.CheckButton] = {}

        for kb_id, label, _hypr_key, description in KEYBIND_PRESETS:
            radio = Gtk.CheckButton(label=label)
            radio.add_css_class("wizard-radio")
            if group is None:
                group = radio
            else:
                radio.set_group(group)

            if kb_id == "f9":
                radio.set_active(True)

            radio.connect("toggled", self._on_keybind_toggled, kb_id)
            page.append(radio)
            self._keybind_radios[kb_id] = radio

            desc_label = Gtk.Label(label=description)
            desc_label.add_css_class("wizard-radio-desc")
            desc_label.set_halign(Gtk.Align.START)
            page.append(desc_label)

        # Live preview of the Hyprland config lines
        self._keybind_preview = Gtk.Label()
        self._keybind_preview.add_css_class("wizard-summary")
        self._keybind_preview.set_halign(Gtk.Align.START)
        self._keybind_preview.set_selectable(True)
        self._update_keybind_preview()
        page.append(self._keybind_preview)

        nav = self._make_nav_row(
            back_page="asr",
            next_page="done",
            next_label="Finish",
        )
        nav.set_margin_top(20)
        page.append(nav)

        return page

    def _build_done_page(self) -> Gtk.Widget:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        page.add_css_class("wizard-page")
        page.set_halign(Gtk.Align.CENTER)
        page.set_valign(Gtk.Align.CENTER)
        page.set_spacing(4)

        self._add_text_title(page, "You\u2019re All Set!")

        sub = Gtk.Label(label="Here\u2019s a summary of your settings:")
        sub.add_css_class("wizard-subtitle")
        page.append(sub)

        self._summary_label = Gtk.Label()
        self._summary_label.add_css_class("wizard-summary")
        self._summary_label.set_halign(Gtk.Align.CENTER)
        page.append(self._summary_label)

        tip = Gtk.Label(
            label="Edit ~/.config/shuvoice/config.toml to adjust ASR and overlay settings.\n"
            "Edit ~/.config/hypr/hyprland.conf to change the push-to-talk keybind."
        )
        tip.add_css_class("wizard-desc")
        tip.set_halign(Gtk.Align.CENTER)
        tip.set_justify(Gtk.Justification.CENTER)
        tip.set_margin_top(16)
        page.append(tip)

        btn = self._make_button("Launch ShuVoice", primary=True)
        btn.connect("clicked", self._on_finish)
        btn.set_halign(Gtk.Align.CENTER)
        btn.set_margin_top(20)
        page.append(btn)

        return page

    # -- Callbacks -------------------------------------------------------------

    def _on_asr_toggled(self, button: Gtk.CheckButton, backend_id: str):
        if button.get_active():
            self._asr_backend = backend_id

    def _on_keybind_toggled(self, button: Gtk.CheckButton, kb_id: str):
        if button.get_active():
            self._keybind = kb_id
            self._update_keybind_preview()

    def _update_keybind_preview(self):
        """Refresh the keybind preview box based on current selection."""
        if not hasattr(self, "_keybind_preview") or self._keybind_preview is None:
            return
        hypr_key = next(
            (hk for kid, _, hk, _ in KEYBIND_PRESETS if kid == self._keybind),
            None,
        )
        if hypr_key:
            bind_text = format_hyprland_bind(hypr_key)
            indented = "\n".join(f"  {l}" for l in bind_text.splitlines())
            text = f"Add to ~/.config/hypr/hyprland.conf:\n\n{indented}"
        else:
            text = (
                "Configure your keybind in ~/.config/hypr/hyprland.conf\n"
                "See README.md for bind/bindr examples."
            )
        self._keybind_preview.set_text(text)

    def _release_input_and_destroy_window(self):
        win = self._win
        if win is None:
            return

        if LayerShell.is_supported():
            try:
                LayerShell.set_keyboard_mode(win, LayerShell.KeyboardMode.NONE)
            except Exception:
                log.debug("Failed to release wizard keyboard mode", exc_info=True)

        try:
            win.set_visible(False)
        except Exception:
            log.debug("Failed to hide wizard window", exc_info=True)

        try:
            win.destroy()
        except Exception:
            log.debug("Failed to destroy wizard window", exc_info=True)

        self._win = None

    def do_shutdown(self):
        self._release_input_and_destroy_window()
        Gtk.Application.do_shutdown(self)

    def _on_finish(self, _button):
        write_config(self._asr_backend)
        write_marker()
        self.completed = True
        log.info("Wizard completed: asr_backend=%s keybind=%s", self._asr_backend, self._keybind)
        self._release_input_and_destroy_window()
        self.quit()

    # -- Helpers ---------------------------------------------------------------

    def _update_summary(self):
        """Refresh the summary label text based on current selections."""
        if not hasattr(self, "_summary_label") or self._summary_label is None:
            return
        self._summary_label.set_text(format_summary(self._asr_backend, self._keybind))

    @staticmethod
    def _add_text_title(box: Gtk.Box, text: str):
        title = Gtk.Label(label=text)
        title.add_css_class("wizard-title")
        title.set_halign(Gtk.Align.CENTER)
        box.append(title)

    @staticmethod
    def _make_button(label: str, *, primary: bool = False) -> Gtk.Button:
        btn = Gtk.Button(label=label)
        btn.add_css_class("wizard-btn")
        if primary:
            btn.add_css_class("wizard-btn-primary")
        return btn

    def _make_nav_row(
        self,
        back_page: str | None = None,
        next_page: str | None = None,
        next_label: str = "Next",
    ) -> Gtk.Box:
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        row.set_halign(Gtk.Align.CENTER)
        row.set_spacing(16)

        if back_page:
            back = self._make_button("Back")
            back.connect("clicked", lambda _: self._stack.set_visible_child_name(back_page))
            row.append(back)

        if next_page:
            nxt = self._make_button(next_label, primary=True)

            def _go_next(_btn, page=next_page):
                if page == "done":
                    self._update_summary()
                self._stack.set_visible_child_name(page)

            nxt.connect("clicked", _go_next)
            row.append(nxt)

        return row
