"""Offline checks for the cost-saving logic. No Windows, game or API key needed:

    python -m unittest discover tests
"""

import json
import os
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from bot.pacing import call_reason, frame_change, frame_signature, should_skip

ROOT = Path(__file__).resolve().parent.parent


def scene(seed=0, h=1080, w=1920):
    rng = np.random.default_rng(seed)
    # Blocky "terrain": 60x60 px tiles of random colours, like a Minecraft view.
    tiles = rng.integers(0, 256, size=(h // 60 + 1, w // 60 + 1, 3), dtype=np.uint8)
    return np.repeat(np.repeat(tiles, 60, axis=0), 60, axis=1)[:h, :w]


class FrameDiff(unittest.TestCase):
    def setUp(self):
        self.base = scene()
        self.sig = frame_signature(self.base)

    def test_signature_shape(self):
        self.assertEqual(self.sig.shape, (18, 32))

    def test_identical_and_noisy_frames_are_static(self):
        self.assertEqual(frame_change(self.sig, frame_signature(self.base.copy())), 0.0)
        noise = np.random.default_rng(1).integers(-6, 7, size=self.base.shape)
        noisy = np.clip(self.base.astype(int) + noise, 0, 255).astype(np.uint8)
        self.assertLess(frame_change(self.sig, frame_signature(noisy)), 3.0)

    def test_crosshair_crack_is_static(self):
        # Mining cracks only change a small patch at the centre.
        cracked = self.base.copy()
        cracked[500:580, 920:1000] = 0
        self.assertLess(frame_change(self.sig, frame_signature(cracked)), 3.0)

    def test_turning_or_walking_changes(self):
        turned = np.roll(self.base, 150, axis=1)  # ~14 degrees of camera turn
        self.assertGreater(frame_change(self.sig, frame_signature(turned)), 3.0)
        self.assertGreater(frame_change(self.sig, frame_signature(scene(seed=2))), 3.0)

    def test_non_contiguous_bgr_view(self):
        bgra = np.dstack([self.base[:, :, ::-1], np.full(self.base.shape[:2], 255, np.uint8)])
        rgb_view = bgra[:, :, 2::-1]  # what Screen.grab returns
        self.assertEqual(frame_change(self.sig, frame_signature(rgb_view)), 0.0)

    def test_missing_signature_counts_as_changed(self):
        self.assertEqual(frame_change(None, self.sig), float("inf"))


class Pacing(unittest.TestCase):
    def test_call_reason(self):
        self.assertEqual(call_reason(True, 5, 0.1, 1.5, 4), "wake")
        self.assertEqual(call_reason(False, 1.0, 1, 1.5, 4), "low")
        self.assertEqual(call_reason(False, 5, 4.5, 1.5, 4), "max")
        self.assertIsNone(call_reason(False, 5, 2, 1.5, 4))
        self.assertIsNone(call_reason(False, 5, 6, 1.5, 10))  # calm: longer max interval

    def test_should_skip(self):
        self.assertFalse(should_skip("wake", 0.0, 0.9, 3, 3))      # alerts always get a call
        self.assertFalse(should_skip("low", 10.0, 0.9, 3, 3))      # screen changed
        self.assertTrue(should_skip("max", 0.5, 20, 3, 3))         # plenty of plan, nothing new
        self.assertTrue(should_skip("low", 0.5, 1.0, 3, 3))        # same picture, wait a bit
        self.assertFalse(should_skip("low", 0.5, 3.5, 3, 3))       # ...but not forever


def usage(i=0, w=0, r=0, o=0):
    return SimpleNamespace(input_tokens=i, cache_creation_input_tokens=w, cache_read_input_tokens=r, output_tokens=o)


class Costs(unittest.TestCase):
    def test_prices_and_components(self):
        from bot.brain import PRICES, CostTracker
        self.assertEqual(PRICES["claude-haiku-5-5"], (0.10, 0.50))
        cfg = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
        for key in ("tactician_model", "strategist_model"):
            self.assertIn(cfg[key], PRICES)
        t = CostTracker(5.0)
        c = t.add("claude-haiku-5-5", usage(i=1000, r=1000, o=200), "tactician")
        self.assertAlmostEqual(c, (1000 * 0.10 + 1000 * 0.01 + 200 * 0.50) / 1e6)
        t.add("claude-opus-5-5", usage(i=1000, w=800, o=500), "strategist")
        t.skip("tactician")
        self.assertEqual(t.parts["tactician"]["skipped"], 1)
        self.assertEqual(t.calls, {"claude-haiku-5-5": 1, "claude-opus-5-5": 1})
        line = t.summary()
        self.assertIn("tactician:", line)
        self.assertIn("strategist:", line)
        self.assertIn("cached", line)


def response(text, stop="end_turn"):
    return SimpleNamespace(stop_reason=stop, usage=usage(i=10, o=10),
                           content=[SimpleNamespace(type="thinking", thinking=""),
                                    SimpleNamespace(type="text", text=text)])


class BrainCalls(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("ANTHROPIC_API_KEY", "test-not-used")
        from bot.brain import Brain, CostTracker
        self.cfg = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
        self.brain = Brain(self.cfg, CostTracker(5.0), log=lambda *_: None)
        self.brain.set_manifesto({"survival_summary": "s", "code_rules": ["c"],
                                  "ambition_summary": "a", "temperament_summary": "t"})

    def test_tactician_request(self):
        plan = {"observation": "o", "plan": [], "note": "n", "objective_complete": False, "stuck": False}
        with mock.patch.object(self.brain.client.messages, "create", return_value=response(json.dumps(plan))) as m:
            result, _ = self.brain.tactics("aGk=", {"objective": "x"})
        self.assertEqual(result, plan)
        kw = m.call_args.kwargs
        self.assertEqual(kw["model"], "claude-haiku-5-5")
        self.assertEqual(kw["thinking"], {"type": "disabled"})
        self.assertEqual(kw["output_config"]["effort"], "low")
        self.assertEqual(kw["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("\n", kw["messages"][0]["content"][1]["text"])  # compact state JSON

    def test_haiku_4_5_gets_no_effort(self):
        self.cfg.update(tactician_model="claude-haiku-4-5", tactician_effort="")
        plan = {"observation": "o", "plan": [], "note": "n", "objective_complete": False, "stuck": False}
        with mock.patch.object(self.brain.client.messages, "create", return_value=response(json.dumps(plan))) as m:
            self.brain.tactics("aGk=", {})
        self.assertNotIn("effort", m.call_args.kwargs["output_config"])

    def test_bad_responses_are_recoverable(self):
        from bot.brain import BadResponse
        for r in (response("{}", stop="refusal"), response('{"plan": [', stop="max_tokens"), response("not json")):
            with mock.patch.object(self.brain.client.messages, "create", return_value=r):
                with self.assertRaises(BadResponse):
                    self.brain.tactics("aGk=", {})


if __name__ == "__main__":
    unittest.main()
