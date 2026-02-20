from __future__ import annotations

import pytest

pytestmark = pytest.mark.gui


def test_gtk_runtime_smoke():
    gi = pytest.importorskip("gi")
    gi.require_version("Gtk", "4.0")
    from gi.repository import Gtk

    window = Gtk.Window()
    assert window is not None
