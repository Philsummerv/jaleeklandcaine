"""Offline checks for the cost-saving logic. No Windows, game or API key needed:

    python -m unittest discover tests
"""

import collections
import json
import os
import threading
import time
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


class HudMasks(unittest.TestCase):
    """The bars are drawn over the world, so terrain colours land inside the boxes."""

    def setUp(self):
        from bot.screen import food_mask, heart_mask
        self.heart, self.food = heart_mask, food_mask

    def px(self, r, g, b):
        return np.array([[[r, g, b]]], dtype=np.int16)

    def test_heart_mask_ignores_badlands_terrain(self):
        self.assertTrue(self.heart(self.px(220, 30, 30))[0, 0])      # heart red
        for name, rgb in (("red sand", (190, 105, 60)), ("terracotta", (152, 94, 67)),
                          ("orange terracotta", (161, 83, 37)), ("grass", (90, 140, 60))):
            self.assertFalse(self.heart(self.px(*rgb))[0, 0], name)

    def test_food_mask_matches_the_drumstick_red_only(self):
        """Measured from raw HUD crops. The icon's brown body is excluded on purpose: it
        is edge pixels blending with the world, and moves with the light behind the bar."""
        for name, rgb in (("red meat", (208, 32, 32)), ("dark red meat", (176, 16, 16))):
            self.assertTrue(self.food(self.px(*rgb))[0, 0], name)
        for name, rgb in (("cooked brown, lit", (160, 112, 80)), ("cooked brown, dim", (144, 96, 64)),
                          ("terracotta behind bar", (192, 112, 64)), ("dark terracotta", (176, 96, 64)),
                          ("red sand", (190, 105, 60)), ("orange terracotta", (161, 83, 37)),
                          ("lava", (255, 120, 0)), ("grass", (90, 140, 60))):
            self.assertFalse(self.food(self.px(*rgb))[0, 0], name)

    def reader(self, **over):
        from bot.screen import HudReader
        cfg = {"health_region": [0, 0, 1, 1], "hunger_region": [0, 0, 1, 1],
               "lava_region": [0, 0, 1, 1], "min_calibration_pixels": 1,
               "max_confirm_frames": 1, "smooth_frames": 5, "max_fall_per_frame": 1.0}
        cfg.update(over)
        return HudReader(cfg)

    @staticmethod
    def bar(filled):
        """A 200px-wide strip with `filled` of it in heart red, the rest terrain."""
        a = np.zeros((10, 200, 3), np.uint8)
        a[:, :] = (96, 48, 32)
        a[:, :filled] = (240, 16, 16)
        return a

    def test_bars_read_proportionally(self):
        """Measured: 2970 px at ten hearts, 1656 at five and a half - 55.8% against 55%."""
        reader = self.reader()
        for _ in range(5):
            reader.read(self.bar(100))
        for _ in range(5):
            reading = reader.read(self.bar(55))
        self.assertAlmostEqual(reading["health"], 0.55, places=2)

    def test_a_hidden_hud_never_reads_as_zero_health(self):
        """A menu, a transition or a hint popup hides the bars. There is no reading to be
        had from those frames, so the last trustworthy one stands: an empty HUD is not a
        dying player, and a dead one gets a death screen rather than empty bars."""
        reader = self.reader()
        for _ in range(5):
            reader.read(self.bar(100))
        blank = np.zeros((10, 200, 3), np.uint8)
        for i in range(4):
            reading = reader.read(blank)
            self.assertEqual(reading["health"], 1.0, f"frame {i + 1} must not invent a reading")
        self.assertFalse(reading["hud_visible"], "but it should say the HUD was unreadable")
        for _ in range(3):      # the median window has to refill before it trusts them again
            back = reader.read(self.bar(55))
        self.assertTrue(back["hud_visible"])
        self.assertAlmostEqual(back["health"], 0.55, places=2)

    def test_cooked_brown_is_not_mistaken_for_a_heart(self):
        # The two bars share a red, which is harmless: each mask only ever sees its own
        # box. What must not happen is the brown of a drumstick reading as health.
        self.assertFalse(self.heart(self.px(144, 96, 64))[0, 0])
        self.assertFalse(self.heart(self.px(160, 112, 80))[0, 0])

    def test_confirmed_max_ignores_a_single_bright_frame(self):
        from bot.screen import HudReader
        cfg = {"health_region": [0, 0, 1, 1], "hunger_region": [0, 0, 1, 1],
               "lava_region": [0, 0, 1, 1], "min_calibration_pixels": 1, "max_confirm_frames": 3}
        reader = HudReader(cfg)
        full = np.zeros((10, 100, 3), np.uint8)
        full[:, :50] = (220, 30, 30)                       # a full bar
        spike = full.copy()
        spike[:, 50:] = (220, 30, 30)                      # one frame of twice as much red
        for _ in range(3):
            reader.read(full)
        base = reader.max_hearts
        reading = reader.read(spike)
        self.assertEqual(reader.max_hearts, base, "one frame must not raise the reference")
        self.assertEqual(reading["health"], 1.0, "health must stay clamped at 100%")
        for _ in range(3):
            reader.read(spike)
        self.assertGreater(reader.max_hearts, base, "a sustained higher reading should count")


class StuckDetection(unittest.TestCase):
    def test_only_walking_counts(self):
        from bot.pacing import looks_stuck
        self.assertTrue(looks_stuck("walk direction=forward seconds=2", 0.2, 1.5))
        self.assertFalse(looks_stuck("walk direction=forward seconds=2", 9.0, 1.5))  # moving fine
        self.assertFalse(looks_stuck("mine seconds=4", 0.2, 1.5))   # mining barely moves the view
        self.assertFalse(looks_stuck("idle", 0.2, 1.5))
        self.assertFalse(looks_stuck("", 0.2, 1.5))

    def test_walking_into_a_wall_reads_as_stuck(self):
        """A wall fills the view, so consecutive frames are near-identical even though the
        bot is holding W. Open ground shifts the whole frame."""
        from bot.pacing import frame_change, frame_signature
        rng = np.random.default_rng(3)
        wall = np.repeat(np.repeat(rng.integers(110, 130, (4, 4, 3), dtype=np.uint8), 300, 0), 500, 1)
        self.assertLess(frame_change(frame_signature(wall), frame_signature(wall.copy())), 1.5)
        moved = np.roll(wall, 120, axis=1)
        self.assertLess(frame_change(frame_signature(wall), frame_signature(moved)), 1.5)  # flat wall


class PlanClamping(unittest.TestCase):
    def test_mining_gets_a_longer_ceiling_than_movement(self):
        """Bare-handed, terracotta needs 6.25s and stone 7.5s. A 6s cap on mine ended the
        hold just before the block gave, and the progress reset to zero every time."""
        from bot.runner import clean_plan
        plan = clean_plan([{"type": "mine", "seconds": 12}, {"type": "walk", "seconds": 12}])
        self.assertEqual(plan[0]["seconds"], 12)
        self.assertEqual(plan[1]["seconds"], 6.0)

    def test_absurd_values_are_still_clamped(self):
        from bot.runner import clean_plan
        plan = clean_plan([{"type": "mine", "seconds": 600}, {"type": "look", "yaw": 9000}])
        self.assertEqual(plan[0]["seconds"], 15.0)
        self.assertEqual(plan[1]["yaw"], 180.0)


class ManifestoReload(unittest.TestCase):
    """The watcher applies an edited manifesto to a running bot without a restart."""

    def setUp(self):
        import tempfile
        from bot.runner import Bot
        self.dir = tempfile.mkdtemp()
        self.path = Path(self.dir) / "manifesto.txt"
        self.write("gather wood")
        self.bot = Bot.__new__(Bot)                     # no window, no API client
        self.bot.cfg = {"manifesto_reload_seconds": 0.01, "rules_cache": str(Path(self.dir) / "r.json")}
        self.bot.running = True
        self.bot.manifesto_path = str(self.path)
        self.bot.manifesto_stamp = None
        self.bot.techniques_path = str(Path(self.dir) / "techniques.txt")
        self.bot.techniques_stamp = None
        self.bot.statements = self.read()
        self.bot.milestones_done = ["old"]
        self.bot.objective = {"objective": "old one"}
        self.bot.objective_history = collections.deque(["old"], maxlen=6)
        self.bot.wake_strategist = threading.Event()
        self.bot.logged = []
        self.bot.log = self.bot.logged.append
        self.compiled = []

    def write(self, ambition):
        self.path.write_text(
            "SURVIVAL: run away when hurt\nCODE: never attack players\n"
            f"AMBITION: {ambition}\nTEMPERAMENT: careful\n", encoding="utf-8")

    def read(self):
        from bot.brain import parse_manifesto
        return parse_manifesto(self.path.read_text(encoding="utf-8"))

    def fake_brain(self, rules):
        brain = type("B", (), {})()
        brain.compile_manifesto = lambda st, cache: (self.compiled.append(st), rules)[1]
        brain.set_manifesto = lambda r: self.compiled.append("applied")
        brain.set_basics = lambda text: self.compiled.append("basics")
        return brain

    def run_watcher(self, edit, until):
        """Start watching, then make the edit: the loop takes the file's stamp as it
        starts, so an edit made beforehand looks like no change at all."""
        t = threading.Thread(target=self.bot.manifesto_loop, daemon=True)
        t.start()
        time.sleep(0.05)
        edit()
        for _ in range(200):
            time.sleep(0.01)
            if until():
                break
        self.bot.running = False
        t.join(timeout=1)

    def test_edit_is_picked_up_and_resets_progress(self):
        new_rules = {"flee_below_health": 0.5, "eat_below_hunger": 0.5, "avoid_lava": True,
                     "code_rules": ["never attack players"], "ambition_milestones": ["a", "b"]}
        self.bot.brain = self.fake_brain(new_rules)
        self.bot.rules = new_rules
        self.run_watcher(lambda: self.write("build a stone tower"), lambda: self.compiled)
        self.assertIn("applied", self.compiled, "new rules should be pushed to the brain")
        self.assertEqual(self.bot.statements["AMBITION"], "build a stone tower")
        self.assertEqual(self.bot.milestones_done, [], "progress belonged to the old ambition")
        self.assertIsNone(self.bot.objective, "the old objective served the old ambition")
        self.assertTrue(self.bot.wake_strategist.is_set(), "should re-plan at once")

    def test_a_broken_manifesto_is_ignored(self):
        self.bot.brain = self.fake_brain({})
        self.bot.rules = {"flee_below_health": 0.5}
        before = dict(self.bot.statements)
        self.run_watcher(lambda: self.path.write_text("SURVIVAL: only this one\n", encoding="utf-8"),
                         lambda: self.bot.logged)
        self.assertEqual(self.bot.statements, before, "a half-written file must not be applied")
        self.assertEqual(self.compiled, [], "and must not be paid for")
        self.assertTrue(any("not usable" in m for m in self.bot.logged))


class ObscuredHud(unittest.TestCase):
    """Measured from lava_trigger.png: a Bedrock hint popup covering the bars read as
    156/2970 health and 0/540 hunger on a player at full health and full hunger."""

    def reader(self):
        from bot.screen import HudReader
        return HudReader({"health_region": [0, 0, 0.5, 1], "hunger_region": [0.5, 0, 1, 1],
                          "lava_region": [0, 0, 1, 1], "min_calibration_pixels": 1,
                          "max_confirm_frames": 1, "smooth_frames": 1, "max_fall_per_frame": 0.05})

    @staticmethod
    def bars(hearts, food):
        a = np.zeros((10, 400, 3), np.uint8)
        a[:, :hearts] = (240, 16, 16)          # health box
        a[:, 200:200 + food] = (208, 32, 32)   # hunger box
        return a

    def test_popup_over_the_bars_is_rejected(self):
        reader = self.reader()
        for _ in range(3):
            reading = reader.read(self.bars(100, 100))
        self.assertEqual(reading["hunger"], 1.0)
        covered = reader.read(self.bars(5, 0))      # the popup frame
        self.assertFalse(covered["hud_visible"])
        self.assertEqual(covered["health"], 1.0, "the last trustworthy reading should stand")
        self.assertEqual(covered["hunger"], 1.0)

    def test_real_hunger_drain_still_gets_through(self):
        reader = self.reader()
        for _ in range(3):
            reader.read(self.bars(100, 100))
        for food in (98, 96, 94, 92, 90):           # a drumstick lost over several frames
            reading = reader.read(self.bars(100, food))
        self.assertTrue(reading["hud_visible"])
        self.assertAlmostEqual(reading["hunger"], 0.90, places=2)

    def test_a_fall_still_reads_as_damage(self):
        """Health may legitimately collapse in one frame; only hunger is the canary."""
        reader = self.reader()
        for _ in range(3):
            reader.read(self.bars(100, 100))
        reading = reader.read(self.bars(10, 100))
        self.assertTrue(reading["hud_visible"])
        self.assertAlmostEqual(reading["health"], 0.10, places=2)


class ManifestoReachesThePrompts(unittest.TestCase):
    """fight_or_flight was compiled, stored, and never passed to either prompt, so nothing
    ever told the bot how to react to a mob. It watched a zombie kill it twice."""

    RULES = {"survival_summary": "s", "code_rules": ["c"], "ambition_summary": "a",
             "temperament_summary": "t", "fight_or_flight": "fight",
             "ambition_milestones": ["8 logs", "a wooden pickaxe"]}

    def brain(self, **over):
        os.environ.setdefault("ANTHROPIC_API_KEY", "test-not-used")
        from bot.brain import Brain, CostTracker
        b = Brain({}, CostTracker(1.0), log=lambda *_: None)
        b.set_manifesto({**self.RULES, **over})
        return b

    def test_stance_reaches_both_prompts(self):
        b = self.brain()
        self.assertIn("fight", b.tactician_system)
        self.assertIn("fight", b.strategist_system)

    def test_milestones_reach_the_strategist(self):
        b = self.brain()
        self.assertIn("8 logs", b.strategist_system)
        self.assertIn("a wooden pickaxe", b.strategist_system)

    def test_nothing_is_left_unformatted(self):
        """A placeholder the rules don't fill would reach Claude as literal braces."""
        b = self.brain()
        for name, text in (("tactician", b.tactician_system), ("strategist", b.strategist_system)):
            self.assertNotIn("{", text, f"{name} prompt has an unfilled placeholder")


class HourlyRate(unittest.TestCase):
    """A one-off compile amortised over the first seconds of a run read as $11.68/hour on
    a run that spent $0.07 in total."""

    def tracker(self, minutes_elapsed):
        from bot.brain import CostTracker
        t = CostTracker(5.0)
        t.add("claude-opus-5-5", usage(i=1500, o=1300), "compile")       # the one-off
        for _ in range(60):
            t.add("claude-haiku-5-5", usage(i=300, r=2700, o=160), "tactician")
        t.started = time.monotonic() - minutes_elapsed * 60
        return t

    def test_no_rate_is_quoted_from_a_few_seconds(self):
        t = self.tracker(10 / 60)
        self.assertIsNone(t.rate_per_hour(10 / 60))
        self.assertIn("rate after a minute", t.summary())

    def test_the_rate_excludes_the_one_off(self):
        t = self.tracker(5.0)
        tactician = t.parts["tactician"]["cost"]
        self.assertAlmostEqual(t.rate_per_hour(5.0), tactician / 5 * 60, places=6)
        self.assertLess(t.rate_per_hour(5.0), t.total / 5 * 60, "the compile must not inflate it")
        self.assertIn("one-off", t.summary(), "but it should still be reported")


class EscapeKey(unittest.TestCase):
    """In Bedrock, escape with no menu open opens the pause menu and freezes the game.
    The bot did that to itself twice in one run, then spent the time reading the menu."""

    def executor(self):
        from bot.executor import Executor
        e = Executor.__new__(Executor)
        e.dry_run = True
        e.menu_open = False
        e.held = set()
        e.buttons = set()
        return e

    def test_escape_is_ignored_with_no_menu_open(self):
        e = self.executor()
        result = e._execute({"type": "key", "key": "escape"})
        self.assertIn("ignored", result)
        self.assertFalse(e.menu_open)

    def test_inventory_opens_then_escape_closes(self):
        e = self.executor()
        self.assertEqual(e._execute({"type": "key", "key": "inventory"}), "ok")
        self.assertTrue(e.menu_open)
        self.assertEqual(e._execute({"type": "key", "key": "escape"}), "ok")
        self.assertFalse(e.menu_open, "and escape is refused again afterwards")
        self.assertIn("ignored", e._execute({"type": "key", "key": "escape"}))

    def test_a_covered_hud_re_enables_escape(self):
        """If something else opened a menu, the runner says so and escape works again."""
        e = self.executor()
        e.menu_may_be_open()
        self.assertEqual(e._execute({"type": "key", "key": "escape"}), "ok")


class TechniquesFile(unittest.TestCase):
    """Technique is handed to Claude as written, so editing it costs nothing to apply."""

    def brain(self):
        os.environ.setdefault("ANTHROPIC_API_KEY", "test-not-used")
        from bot.brain import Brain, CostTracker
        b = Brain({}, CostTracker(1.0), log=lambda *_: None)
        b.set_manifesto({"survival_summary": "s", "code_rules": ["c"], "ambition_summary": "a",
                         "temperament_summary": "t", "fight_or_flight": "fight",
                         "ambition_milestones": ["8 logs"]})
        return b

    def test_techniques_reach_both_prompts(self):
        b = self.brain()
        b.set_basics("- To climb a one-block step, walk forward with jump held.")
        self.assertIn("walk forward with jump held", b.tactician_system)
        self.assertIn("walk forward with jump held", b.strategist_system)

    def test_an_edit_applies_without_recompiling(self):
        """set_basics re-renders the prompts from the rules it already has, so a technique
        change needs no compile call and no new rules."""
        b = self.brain()
        b.set_basics("first version")
        b.set_basics("second version")
        self.assertIn("second version", b.tactician_system)
        self.assertNotIn("first version", b.tactician_system)

    def test_the_shipped_file_is_loadable_and_substantial(self):
        text = (ROOT / "techniques.txt").read_text(encoding="utf-8")
        body = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
        self.assertGreater(len(body), 40, "the shipped playbook should actually say something")
        b = self.brain()
        b.set_basics(text)
        self.assertNotIn("{", b.tactician_system, "no placeholder left unfilled")


class DamageConfirmation(unittest.TestCase):
    """Readings were seen swinging 39% -> 81% -> 36% inside one second, each bounce
    announced as an attack. Health cannot do that."""

    @staticmethod
    def alerts_for(readings, spread=0.12, threshold=0.04):
        """Replay health readings through the damage rule, as the reflex loop does."""
        from bot.pacing import settled_damage
        recent, baseline, alerts = collections.deque(maxlen=3), None, []
        for health in readings:
            recent.append(health)
            hit = settled_damage(recent, baseline, threshold, spread)
            if hit is not None:
                baseline = hit
                alerts.append(round(hit, 2))
            elif baseline is None or health > baseline:
                baseline = health
        return alerts

    def test_a_bouncing_reading_raises_nothing(self):
        """The readings from the run that prompted this, verbatim."""
        self.assertEqual(self.alerts_for([1.0, 0.39, 0.81, 0.36, 0.85, 0.39, 0.81]), [])

    def test_a_hit_that_settles_registers(self):
        """A zombie takes a bite: the bar steps down and holds there."""
        self.assertEqual(self.alerts_for([1.0, 1.0, 1.0, 0.86, 0.86, 0.86]), [0.86])

    def test_several_hits_each_register(self):
        readings = [1.0] * 3 + [0.86] * 3 + [0.70] * 3 + [0.56] * 3
        self.assertEqual(self.alerts_for(readings), [0.86, 0.70, 0.56])

    def test_healing_raises_nothing(self):
        self.assertEqual(self.alerts_for([0.4] * 3 + [0.6] * 3 + [0.8] * 3 + [1.0] * 3), [])

    def test_a_single_bad_frame_raises_nothing(self):
        self.assertEqual(self.alerts_for([1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0]), [])
