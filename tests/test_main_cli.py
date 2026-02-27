from __future__ import annotations

from argparse import Namespace

import pytest

from shuvoice.__main__ import _apply_cli_overrides
from shuvoice.cli.parser import create_parser, resolve_command
from shuvoice.config import Config


def _args(**overrides) -> Namespace:
    values = {
        "asr_backend": None,
        "device": None,
        "right_context": None,
        "sherpa_model_dir": None,
        "sherpa_model_name": None,
        "sherpa_provider": None,
        "sherpa_num_threads": None,
        "sherpa_chunk_ms": None,
        "moonshine_model_name": None,
        "moonshine_model_dir": None,
        "moonshine_model_precision": None,
        "moonshine_chunk_ms": None,
        "moonshine_max_window_sec": None,
        "moonshine_max_tokens": None,
        "moonshine_provider": None,
        "moonshine_onnx_threads": None,
        "audio_device": None,
        "input_gain": None,
        "output_mode": None,
        "control_socket": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _parse(argv: list[str]):
    parser = create_parser()
    args = parser.parse_args(argv)
    route, warnings = resolve_command(args, parser)
    return parser, args, route, warnings


def test_apply_cli_overrides_moonshine_provider_and_threads() -> None:
    config = Config()

    _apply_cli_overrides(
        _args(
            asr_backend="moonshine",
            moonshine_provider="cuda",
            moonshine_onnx_threads=4,
            audio_device="2",
        ),
        config,
    )

    assert config.asr_backend == "moonshine"
    assert config.moonshine_provider == "cuda"
    assert config.moonshine_onnx_threads == 4
    assert config.audio_device == 2


def test_apply_cli_overrides_sherpa_model_name() -> None:
    config = Config()

    _apply_cli_overrides(
        _args(
            asr_backend="sherpa",
            sherpa_model_name="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
        ),
        config,
    )

    assert config.asr_backend == "sherpa"
    assert config.sherpa_model_name == "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"


def test_default_route_is_run():
    _parser, _args, route, warnings = _parse([])
    assert route == "run"
    assert warnings == []


def test_legacy_preflight_maps_to_preflight_route():
    _parser, _args, route, warnings = _parse(["--preflight"])
    assert route == "preflight"
    assert warnings


def test_legacy_control_maps_to_control_route_and_action():
    _parser, args, route, warnings = _parse(["--control", "status"])
    assert route == "control"
    assert args.control_action == "status"
    assert warnings


def test_subcommand_control_maps_without_legacy_warning():
    _parser, args, route, warnings = _parse(["control", "status"])
    assert route == "control"
    assert args.control_action == "status"
    assert warnings == []


def test_subcommand_config_effective_route():
    _parser, _args, route, _warnings = _parse(["config", "effective"])
    assert route == "config_effective"


def test_subcommand_config_validate_route():
    _parser, _args, route, _warnings = _parse(["config", "validate"])
    assert route == "config_validate"


def test_subcommand_config_set_route():
    _parser, _args, route, _warnings = _parse(
        ["config", "set", "typing_final_injection_mode", "direct"]
    )
    assert route == "config_set"


def test_subcommand_model_download_route():
    _parser, _args, route, _warnings = _parse(["model", "download"])
    assert route == "model_download"


def test_subcommand_audio_list_devices_route():
    _parser, _args, route, _warnings = _parse(["audio", "list-devices"])
    assert route == "audio_list_devices"


def test_subcommand_diagnostics_route():
    _parser, _args, route, _warnings = _parse(["diagnostics"])
    assert route == "diagnostics"


def test_subcommand_setup_route():
    _parser, _args, route, _warnings = _parse(["setup"])
    assert route == "setup"


def test_legacy_flags_are_mutually_exclusive():
    parser = create_parser()
    args = parser.parse_args(["--preflight", "--wizard"])
    with pytest.raises(SystemExit):
        resolve_command(args, parser)
