# ovos-eot-plugin-cuelight

CueLight end-of-turn plugin for [OpenVoiceOS](https://openvoiceos.com).

[CueLight](https://github.com/sshah03/CueLight) is an always-listening turn
detector. It gives a new end-of-turn probability every 20 ms of audio,
including while the user still speaks. The code and weights are Apache-2.0.
CueLight is English only.

This plugin wraps CueLight behind the OpenVoiceOS `opm.end_of_turn` plugin
type, so a listener ends a voice command when the user finished speaking
instead of waiting for a fixed silence timer.

## Install

This plugin needs the `cuelight` package from the upstream project, which is
not on PyPI yet:

```bash
pip install git+https://github.com/sshah03/CueLight
pip install ovos-eot-plugin-cuelight
```

The model weights download from Hugging Face the first time the plugin loads
them, into the shared Hugging Face cache.

## Configuration

Add an `end_of_turn` block to the `listener` section of `mycroft.conf`:

```json
{
  "listener": {
    "end_of_turn": {
      "module": "ovos-eot-plugin-cuelight",
      "ovos-eot-plugin-cuelight": {
        "model": "cuelight-nano",
        "preset": "snappy"
      }
    }
  }
}
```

| Key | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `cuelight-nano` | `cuelight-nano`, `cuelight-base`, or a local folder |
| `preset` | `str` | `snappy` | `snappy` responds sooner. `careful` interrupts less |

When the config sets `threshold`, `min_silence_s`, or `timeout_s`, the plugin
builds a custom `EndpointPolicy` and ignores `preset`.

| Key | Type | Default | Description |
|---|---|---|---|
| `threshold` | `float` | - | p_end above which the turn may end |
| `min_silence_s` | `float` | - | silence seconds before `threshold` applies |
| `timeout_s` | `float` | - | silence seconds after which the turn always ends |
| `reset_on_turn_start` | `bool` | `true` | let CueLight reset itself when a new turn starts |

`cuelight-nano` is 1.9 MB and costs about 0.5 ms per 20 ms of audio.
`cuelight-base` is 336 MB and keeps about a quarter of one CPU core busy. Use
it on servers and desktops.

## Models

| model | parameters | file size | best for |
|---|---:|---:|---|
| CueLight-Nano | 0.5M | 1.9 MB | phones, wearables, embedded devices |
| CueLight-Base | 88M | 336 MB | servers and desktops |

Only Nano has an ONNX export upstream. This plugin uses the PyTorch weights.

## License

Apache-2.0. CueLight is a separate Apache-2.0 project and its own notices
apply to the code and weights it ships.
