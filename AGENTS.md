# ovos-eot-plugin-cuelight

OVOS end-of-turn plugin wrapping the CueLight streaming turn detector. Implements the `opm.end_of_turn` plugin type so a listener ends a voice command when the user finished speaking, instead of waiting for a fixed silence timer.

## Setup

```bash
pip install -e .
```

Runtime deps: `ovos-plugin-manager>=3.0.0a1` (the `opm.end_of_turn` type and the `EndOfTurnEngine` template), `numpy<3.0.0`. The `cuelight` package is a soft dependency: the plugin imports it lazily and raises a clear `ImportError` when it is missing. Install it from upstream, which is not on PyPI yet:

```bash
pip install git+https://github.com/sshah03/CueLight
```

## Test

```bash
pytest test
```

The tests inject a fake `cuelight` module, so they run without the upstream package installed. They need `ovos-plugin-manager` with the `opm.end_of_turn` type on the path.

## Lint/Typecheck

None configured in-repo. Linting runs through the shared `lint.yml` workflow.

## Layout

- `ovos_eot_plugin_cuelight/__init__.py` — all logic:
  - `_import_cuelight()` — lazy import of `Endpointer`, `EndpointPolicy`, `TurnDetector`.
  - `CueLightEndOfTurn(EndOfTurnEngine)` — the plugin. Reads `model`, `preset`, `threshold`, `min_silence_s`, `timeout_s`, and `reset_on_turn_start` from config. Re-frames 16-bit PCM chunks into 20 ms float32 frames and calls `Endpointer.push` per frame.
- `ovos_eot_plugin_cuelight/version.py` — semver block; never edit by hand.

Entry-point group (declared in `pyproject.toml`):
- `opm.end_of_turn` → `ovos-eot-plugin-cuelight=ovos_eot_plugin_cuelight:CueLightEndOfTurn`

## Conventions

- Branches: `dev` (work) and `master` (stable). NEVER `main`.
- Never edit `version.py`; gh-automations bump semver from conventional-commit prefixes (`feat:`/`fix:`/`feat!:`).
- New repos private by default.
- Commit identity: JarbasAi <jarbasai@mailfence.com>.
- Reference `OpenVoiceOS/gh-automations` reusable workflows at `@dev`.
- No Neon / `neon-*` references.
- No meta-commentary (no history, no dates) in docs/commits/code.
- CI is provided by OpenVoiceOS/gh-automations.

## Gotchas

- The plugin requires 16 kHz audio; any other sample rate raises `ValueError` in `__init__`.
- CueLight is English only; `supported_langs` is `("en",)`.
- `cuelight` is not on PyPI; do not declare it as a git-URL dependency. Install it separately.
- Only CueLight-Nano has an ONNX export upstream; this plugin uses the PyTorch weights.
- `p_end` and `p_speech` are read off the CueLight endpointer/detector; when a future CueLight version drops an attribute, the property returns `None` or `0.0` rather than raising.
