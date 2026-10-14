"""CueLight end-of-turn plugin for OpenVoiceOS.

CueLight (https://github.com/sshah03/CueLight) is an always-listening turn
detector. It gives a new end-of-turn probability every 20 ms of audio,
including while the user still speaks. It is English only.

This plugin wraps CueLight's :class:`cuelight.Endpointer` behind the OpenVoiceOS
:class:`~ovos_plugin_manager.templates.eot.EndOfTurnEngine` contract, so a
listener can end a voice command when the user finished speaking.
"""

from __future__ import annotations

import numpy as np
from ovos_plugin_manager.templates.eot import EndOfTurnDecision, EndOfTurnEngine

#: CueLight re-frames internally, so input chunks of any length are accepted,
#: but its model runs on 20 ms frames at 16 kHz.
FRAME_S = 0.02
CUELIGHT_SAMPLE_RATE = 16000

_CUELIGHT_MISSING = (
    "the 'cuelight' package is not installed; install it from the upstream "
    "project: pip install git+https://github.com/sshah03/CueLight"
)


def _import_cuelight():
    """Import CueLight lazily so the plugin loads without it installed.

    Returns:
        tuple: ``(Endpointer, EndpointPolicy, TurnDetector)``.

    Raises:
        ImportError: when the ``cuelight`` package is missing.
    """
    try:
        from cuelight import Endpointer, EndpointPolicy, TurnDetector
    except ImportError as err:
        raise ImportError(_CUELIGHT_MISSING) from err
    return Endpointer, EndpointPolicy, TurnDetector


class CueLightEndOfTurn(EndOfTurnEngine):
    """End-of-turn engine backed by CueLight.

    Recognized config keys:

    - ``model`` (str): ``"cuelight-nano"`` (default) or ``"cuelight-base"``,
      or a local folder with ``config.json`` and ``model.safetensors``.
    - ``preset`` (str): ``"snappy"`` (default) or ``"careful"``. Ignored when
      a custom policy is set.
    - ``threshold`` (float): p_end above which the turn may end. Setting any
      of ``threshold``, ``min_silence_s``, or ``timeout_s`` builds a custom
      ``EndpointPolicy`` and ignores ``preset``.
    - ``min_silence_s`` (float): silence seconds before the threshold applies.
    - ``timeout_s`` (float): silence seconds after which the turn always ends.
    - ``reset_on_turn_start`` (bool): let CueLight reset itself when a new turn
      starts (default ``True``). Set ``False`` when the listener already
      restarts the engine at each turn.
    """

    supported_langs: tuple[str, ...] = ("en",)

    def __init__(self, config: dict | None = None,
                 sample_rate: int | None = None):
        super().__init__(config, sample_rate)
        if self.sample_rate != CUELIGHT_SAMPLE_RATE:
            raise ValueError(
                f"CueLight needs {CUELIGHT_SAMPLE_RATE} Hz audio, got "
                f"{self.sample_rate} Hz"
            )
        self.model = self.config.get("model", "cuelight-nano")
        self._frame_length = int(self.sample_rate * FRAME_S)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._ep = self._load_endpointer()

    def _load_endpointer(self):
        """Build the CueLight endpointer from config."""
        endpointer, policy_cls, detector_cls = _import_cuelight()
        reset_on_turn_start = self.config.get("reset_on_turn_start", True)

        if self._has_custom_policy():
            detector = detector_cls.from_pretrained(
                self.model, reset_on_turn_start=reset_on_turn_start
            )
            return endpointer(detector, self._build_policy(policy_cls))

        return endpointer.from_pretrained(
            self.model,
            preset=self.config.get("preset", "snappy"),
            reset_on_turn_start=reset_on_turn_start,
        )

    def _has_custom_policy(self) -> bool:
        """True when the config asks for an explicit endpoint policy."""
        return any(
            key in self.config
            for key in ("threshold", "min_silence_s", "timeout_s")
        )

    def _build_policy(self, policy_cls):
        """Build an ``EndpointPolicy`` from the config keys that are set."""
        kwargs = {
            key: self.config[key]
            for key in ("threshold", "min_silence_s", "timeout_s")
            if key in self.config
        }
        return policy_cls(**kwargs)

    def _frames(self, chunk: bytes):
        """Yield 20 ms float32 frames from an int16 PCM chunk."""
        audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        self._buffer = np.concatenate([self._buffer, audio])
        while len(self._buffer) >= self._frame_length:
            frame = self._buffer[:self._frame_length]
            self._buffer = self._buffer[self._frame_length:]
            yield frame

    def update(self, chunk: bytes) -> EndOfTurnDecision:
        """Feed a chunk and return the end-of-turn decision.

        Parameters:
            chunk (bytes): 16-bit PCM, mono, 16 kHz. Any length.

        Returns:
            EndOfTurnDecision: the decision for this chunk. ``ended`` is True
                when CueLight fired on any frame in the chunk.
        """
        ended = False
        for frame in self._frames(chunk):
            ended = bool(self._ep.push(frame)) or ended
        return EndOfTurnDecision(
            p_end=self.p_end,
            p_speech=self.p_speech,
            ended=ended,
        )

    def get_probability(self, chunk: bytes) -> float:
        """Feed a chunk and return only the current end-of-turn probability."""
        return self.update(chunk).p_end

    @property
    def p_end(self) -> float:
        """Probability the user finished their turn, for the last frame."""
        return float(getattr(self._ep, "p_end", 0.0))

    @property
    def p_speech(self) -> float | None:
        """Probability the user is speaking, if the engine reports it."""
        detector = getattr(self._ep, "det", None)
        value = getattr(detector, "p_speech", None)
        return None if value is None else float(value)

    def reset(self):
        """Reset the engine for a new user turn."""
        self._buffer = np.zeros(0, dtype=np.float32)
        detector = getattr(self._ep, "det", None)
        if detector is not None and hasattr(detector, "clear"):
            detector.clear()
        # the endpointer starts with nothing to end until the user speaks again
        if hasattr(self._ep, "silent_frames"):
            self._ep.silent_frames = 0
        if hasattr(self._ep, "fired"):
            self._ep.fired = True
        if hasattr(self._ep, "p_end"):
            self._ep.p_end = 0.0
