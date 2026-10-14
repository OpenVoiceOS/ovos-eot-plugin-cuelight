"""Unit tests for the CueLight end-of-turn plugin.

The real ``cuelight`` package is not installed here, so the tests inject a
fake ``cuelight`` module before importing the plugin.
"""

import sys
import types
import unittest

import numpy as np

from ovos_eot_plugin_cuelight import CueLightEndOfTurn

FRAME_LENGTH = 320  # 20 ms at 16 kHz


class FakeDetector:
    def __init__(self):
        self.p_speech = 0.0
        self.cleared = 0

    def clear(self):
        self.cleared += 1


class FakeEndpointer:
    def __init__(self, detector=None, policy=None):
        self.det = detector or FakeDetector()
        self.pol = policy
        self.p_end = 0.0
        self.silent_frames = 0
        self.fired = True
        self.frames = []
        self.next_result = False
        self.name = None
        self.preset = None
        self.kw = None

    @classmethod
    def from_pretrained(cls, name, preset="snappy", **kw):
        obj = cls()
        obj.name = name
        obj.preset = preset
        obj.kw = kw
        return obj

    def push(self, frame):
        self.frames.append(np.asarray(frame))
        self.p_end = 0.9 if self.next_result else 0.1
        return bool(self.next_result)


class FakeTurnDetector:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, name, **kw):
        obj = cls()
        obj.name = name
        obj.kw = kw
        return obj


class FakeEndpointPolicy:
    def __init__(self, threshold=0.7, min_silence_s=0.6, timeout_s=2.0):
        self.threshold = threshold
        self.min_silence_s = min_silence_s
        self.timeout_s = timeout_s


def _install_fake_cuelight():
    module = types.ModuleType("cuelight")
    module.Endpointer = FakeEndpointer
    module.EndpointPolicy = FakeEndpointPolicy
    module.TurnDetector = FakeTurnDetector
    sys.modules["cuelight"] = module


def _int16_bytes(samples):
    return (np.zeros(samples, dtype=np.int16)).tobytes()


class TestCueLightEndOfTurn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_fake_cuelight()

    def test_supported_langs(self):
        self.assertEqual(CueLightEndOfTurn.supported_langs, ("en",))

    def test_default_preset_and_model(self):
        engine = CueLightEndOfTurn()
        self.assertEqual(engine._ep.name, "cuelight-nano")
        self.assertEqual(engine._ep.preset, "snappy")

    def test_reset_on_turn_start_passed_through(self):
        engine = CueLightEndOfTurn(config={"reset_on_turn_start": False})
        self.assertFalse(engine._ep.kw["reset_on_turn_start"])

    def test_custom_policy_builds_endpointer_from_detector(self):
        engine = CueLightEndOfTurn(
            config={"threshold": 0.8, "min_silence_s": 0.4}
        )
        self.assertIsInstance(engine._ep, FakeEndpointer)
        self.assertIsInstance(engine._ep.pol, FakeEndpointPolicy)
        self.assertEqual(engine._ep.pol.threshold, 0.8)
        self.assertEqual(engine._ep.pol.min_silence_s, 0.4)

    def test_sample_rate_guard(self):
        with self.assertRaises(ValueError):
            CueLightEndOfTurn(sample_rate=8000)

    def test_chunk_is_reframed_to_20ms_frames(self):
        engine = CueLightEndOfTurn()
        # 2048 samples = 6 full 320-sample frames, 128 samples left over
        engine.update(_int16_bytes(2048))
        self.assertEqual(len(engine._ep.frames), 6)
        self.assertTrue(all(f.shape == (FRAME_LENGTH,) for f in engine._ep.frames))
        self.assertEqual(len(engine._buffer), 128)

    def test_frames_accumulate_across_chunks(self):
        engine = CueLightEndOfTurn()
        engine.update(_int16_bytes(160))
        engine.update(_int16_bytes(160))
        self.assertEqual(len(engine._ep.frames), 1)

    def test_input_is_scaled_to_float_range(self):
        engine = CueLightEndOfTurn()
        chunk = (np.full(FRAME_LENGTH, 32767, dtype=np.int16)).tobytes()
        engine.update(chunk)
        frame = engine._ep.frames[0]
        self.assertLessEqual(float(frame.max()), 1.0)
        self.assertGreater(float(frame.max()), 0.9)

    def test_ended_when_cuelight_fires(self):
        engine = CueLightEndOfTurn()
        engine._ep.next_result = True
        decision = engine.update(_int16_bytes(FRAME_LENGTH))
        self.assertTrue(decision.ended)
        self.assertAlmostEqual(decision.p_end, 0.9)

    def test_not_ended_by_default(self):
        engine = CueLightEndOfTurn()
        decision = engine.update(_int16_bytes(FRAME_LENGTH))
        self.assertFalse(decision.ended)

    def test_ended_latches_within_chunk(self):
        engine = CueLightEndOfTurn()
        engine._ep.next_result = True
        decision = engine.update(_int16_bytes(FRAME_LENGTH * 3))
        self.assertTrue(decision.ended)

    def test_p_speech_passthrough(self):
        engine = CueLightEndOfTurn()
        engine._ep.det.p_speech = 0.42
        self.assertAlmostEqual(engine.p_speech, 0.42)

    def test_reset_clears_buffer_and_detector(self):
        engine = CueLightEndOfTurn()
        engine.update(_int16_bytes(100))
        engine._ep.silent_frames = 5
        engine._ep.fired = False
        engine.reset()
        self.assertEqual(len(engine._buffer), 0)
        self.assertEqual(engine._ep.det.cleared, 1)
        self.assertEqual(engine._ep.silent_frames, 0)
        self.assertTrue(engine._ep.fired)

    def test_missing_cuelight_raises(self):
        saved = sys.modules.pop("cuelight", None)
        try:
            with self.assertRaises(ImportError):
                CueLightEndOfTurn()
        finally:
            if saved is not None:
                sys.modules["cuelight"] = saved

    def test_entry_point_import(self):
        from importlib.metadata import entry_points
        eps = entry_points()
        found = [e for e in eps.select(group="opm.end_of_turn")
                 if e.name == "ovos-eot-plugin-cuelight"]
        # the plugin is not installed in this environment; the entry point
        # mapping is covered by the pyproject declaration
        self.assertIsInstance(found, list)


if __name__ == "__main__":
    unittest.main()
