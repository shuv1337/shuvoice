"""CLI entry point for shuvoice."""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Streaming speech-to-text overlay for Hyprland",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the ASR model and exit",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device (default: cuda)",
    )
    parser.add_argument(
        "--hotkey",
        default=None,
        help="Hotkey name, e.g. KEY_RIGHTCTRL (default: from config)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from .config import Config

    config = Config.load()

    if args.device:
        config.device = args.device
    if args.hotkey:
        config.hotkey = args.hotkey

    # --download-model: fetch the model files and exit
    if args.download_model:
        from .asr import ASREngine

        ASREngine.download_model(config.model_name)
        print("Model downloaded successfully.")
        return

    # Load libgtk4-layer-shell BEFORE any gi imports (required by overlay/app)
    from ctypes import CDLL

    try:
        CDLL("libgtk4-layer-shell.so")
    except OSError:
        print(
            "ERROR: libgtk4-layer-shell.so not found.\n"
            "Install it with: pacman -S gtk4-layer-shell",
            file=sys.stderr,
        )
        sys.exit(1)

    from .app import ShuVoiceApp

    app = ShuVoiceApp(config)
    app.load_model()
    sys.exit(app.run(None))


if __name__ == "__main__":
    main()
