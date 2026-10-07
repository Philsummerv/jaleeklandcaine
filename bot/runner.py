"""Runs the bot: reflexes (local, 10 Hz), tactician (Claude, every 1-8 s),
strategist (Claude, when an objective ends or gets stuck) and the executor, all in parallel."""

import collections
import threading
import time
import traceback

import anthropic

from . import winapi
from .brain import BadResponse, Brain, BudgetExceeded, CostTracker
from .executor import Executor
from .pacing import call_reason, frame_change, frame_signature, should_skip
from .screen import HudReader, Screen, encode_jpeg

MAX_SECONDS = 6.0


def clean_plan(plan):
    """Clamp model output to sane ranges so one weird value can't spin the camera 50 times."""
    out = []
    for a in plan[:10]:
        a = {k: v for k, v in a.items() if v is not None}
        if "seconds" in a:
            a["seconds"] = max(0.05, min(float(a["seconds"]), MAX_SECONDS))
        if "yaw" in a:
            a["yaw"] = max(-180.0, min(float(a["yaw"]), 180.0))
        if "pitch" in a:
            a["pitch"] = max(-90.0, min(float(a["pitch"]), 90.0))
        if "times" in a:
            a["times"] = max(1, min(int(a["times"]), 10))
        if a["type"] == "gui_click" and ("x" not in a or "y" not in a):
            continue
        out.append(a)
    return out


class Bot:
    def __init__(self, cfg, statements, dry_run=False):
        self.cfg = cfg
        self.statements = statements
        self.dry_run = dry_run
        self.paused = True
        self.running = True
        self.log_file = open(cfg.get("log_file", "bot.log"), "a", encoding="utf-8")

        self.screen = Screen(cfg["window_title"])
        if not self.screen.locate():
            raise RuntimeError(f"Can't find a window titled '{cfg['window_title']}'. Start Minecraft first.")
        self.hud_reader = HudReader(cfg["hud"])
        self.costs = CostTracker(cfg["max_dollars_per_session"])
        self.brain = Brain(cfg, self.costs, log=self.log)
        self.executor = Executor(self.screen, cfg, self.is_active, dry_run=dry_run, log=self.log)

        self.hud = {"health": None, "hunger": None, "lava": 0.0}
        self.alerts = collections.deque(maxlen=6)
        self.objective = None
        self.objective_started = 0.0
        self.objective_history = collections.deque(maxlen=6)
        self.tactician_note = ""
        self.last_observation = ""
        self.wake_tactician = threading.Event()
        self.wake_strategist = threading.Event()
        self.last_damage = 0.0
        self.last_stuck = 0.0

    # --- helpers ----------------------------------------------------------

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def is_active(self):
        return self.running and not self.paused and self.screen.focused()

    def alert(self, text):
        self.alerts.append((time.monotonic(), text))
        self.log(f"!! {text}")
        self.wake_tactician.set()

    def recent_alerts(self, within=8.0):
        now = time.monotonic()
        return [f"{now - t:.0f}s ago: {msg}" for t, msg in self.alerts if now - t < within]

    def vitals(self):
        h, f = self.hud.get("health"), self.hud.get("hunger")
        return {
            "health": f"{h:.0%}" if h is not None else "unknown",
            "hunger": f"{f:.0%}" if f is not None else "unknown",
        }

    def frame_b64(self, frame=None):
        if frame is None:
            frame = self.screen.grab()
        return encode_jpeg(frame, self.cfg["screenshot_width"], self.cfg.get("jpeg_quality", 70))

    def calm(self):
        """True when nothing needs fast reactions: no recent alerts, reflexes or stuck reports,
        and the objective isn't brand new."""
        now = time.monotonic()
        return (not self.recent_alerts()
                and not self.executor.in_reflex()
                and now - self.last_stuck > 10
                and now - self.objective_started > 5)

    def _spawn(self, fn):
        def guarded():
            try:
                fn()
            except BudgetExceeded as e:
                self.log(f"Stopping: {e}")
                self.running = False
            except Exception:
                self.log("Thread crashed:\n" + traceback.format_exc())
                self.running = False
        t = threading.Thread(target=guarded, daemon=True)
        t.start()
        return t

    # --- reflexes: local pixel reading, no API -----------------------------

    def reflex_loop(self):
        rules = self.rules
        prev_health = None
        flee_cooldown = lava_cooldown = hunger_nag = 0.0
        while self.running:
            time.sleep(0.1)
            try:
                frame = self.screen.grab()
            except Exception:
                continue
            hud = self.hud_reader.read(frame)
            self.hud = hud
            if not self.is_active():
                prev_health = hud["health"]
                continue
            now = time.monotonic()
            health = hud["health"]

            if health is not None and prev_health is not None and health < prev_health - 0.04:
                self.last_damage = now
                self.alert(f"took damage, health now {health:.0%}")
            prev_health = health

            if (health is not None and rules["flee_below_health"] > 0 and health < rules["flee_below_health"]
                    and now - self.last_damage < 3 and now > flee_cooldown and not self.executor.in_reflex()):
                self.executor.set_plan([
                    {"type": "look", "yaw": 180, "pitch": 0},
                    {"type": "walk", "direction": "forward", "seconds": 2.5, "sprint": True, "jump": True},
                ], source="reflex")
                flee_cooldown = now + 6
                self.alert("REFLEX: low health under attack, fleeing")
                self.wake_strategist.set()  # SURVIVAL may call for a different objective

            if rules["avoid_lava"] and hud["lava"] > self.cfg["hud"]["lava_fraction"] and now > lava_cooldown:
                self.executor.set_plan([{"type": "walk", "direction": "back", "seconds": 0.6}], source="reflex")
                lava_cooldown = now + 2
                self.alert("REFLEX: lava ahead, backed off")

            hunger = hud["hunger"]
            if hunger is not None and hunger < rules["eat_below_hunger"] and now > hunger_nag:
                hunger_nag = now + 20
                self.alert(f"hungry ({hunger:.0%}): eat food if you have any (select it in the hotbar, use ~1.8s)")

    # --- strategist: slow, decides WHAT to do ------------------------------

    def strategist_loop(self):
        # Called when there's no objective, when the tactician reports it complete or stuck,
        # after a flee reflex, and otherwise every `interval` seconds as a fallback.
        interval = self.cfg["strategist_interval_seconds"]
        last = 0.0
        while self.running:
            due = self.objective is None or time.monotonic() - last > interval or self.wake_strategist.is_set()
            if not (due and self.is_active()):
                time.sleep(0.2)
                continue
            self.wake_strategist.clear()
            last = time.monotonic()
            state = {
                "current_objective": self.objective,
                "minutes_on_current_objective": round((time.monotonic() - self.objective_started) / 60, 1) if self.objective else 0,
                "previous_objectives": list(self.objective_history),
                "tactician_latest_observation": self.last_observation,
                "tactician_note": self.tactician_note,
                "vitals": self.vitals(),
                "recent_alerts": self.recent_alerts(30),
            }
            try:
                result, cost = self.brain.strategize(self.frame_b64(), state)
            except (anthropic.APIError, BadResponse) as e:
                self.log(f"strategist error: {e}")
                time.sleep(3)
                continue
            old = self.objective["objective"] if self.objective else None
            if old != result["objective"]:
                if old and not (self.objective_history and self.objective_history[-1].startswith(old)):
                    self.objective_history.append(f"{old} -> replaced")
                self.objective_started = time.monotonic()
                self.tactician_note = ""
            self.objective = result
            self.log(f"STRATEGY (${cost:.3f}): {result['objective']}  |  {result['situation']}")
            self.wake_tactician.set()

    # --- tactician: fast, decides HOW (button presses) ---------------------

    def tactician_loop(self):
        c = self.cfg
        adaptive = c.get("tactician_pacing", "fixed") == "adaptive"
        skip_static = c.get("tactician_skip_static_frames", False)
        last = 0.0
        last_sig = None
        recheck_at = 0.0
        skipping = False
        while self.running:
            time.sleep(0.03)
            if not self.is_active() or self.objective is None or self.executor.in_reflex():
                continue
            now = time.monotonic()
            since = now - last
            if since < c["tactician_min_interval"] or (now < recheck_at and not self.wake_tactician.is_set()):
                continue
            calm = adaptive and self.calm()
            reason = call_reason(
                self.wake_tactician.is_set(),
                self.executor.remaining_seconds(),
                since,
                c["tactician_lookahead_seconds"],
                c["tactician_calm_max_interval"] if calm else c["tactician_max_interval"])
            if reason is None:
                continue
            frame = self.screen.grab()
            sig = frame_signature(frame) if skip_static else None
            if skip_static and should_skip(reason, frame_change(sig, last_sig), since,
                                           c["tactician_static_threshold"], c["tactician_static_interval"]):
                if not skipping:  # count each held-back call once, not every recheck
                    self.costs.skip("tactician")
                    skipping = True
                recheck_at = now + 0.25
                continue
            skipping = False
            self.wake_tactician.clear()
            last = time.monotonic()
            last_sig = sig
            if adaptive:
                lo, hi = c["tactician_calm_plan_seconds"] if calm else c["tactician_urgent_plan_seconds"]
            else:
                lo, hi = 2, 5
            state = {
                "objective": self.objective["objective"],
                "done_when": self.objective["done_when"],
                "strategist_hints": self.objective["hints"],
                "plan_seconds": f"{lo:g}-{hi:g}",
                "vitals": self.vitals(),
                "alerts": self.recent_alerts(),
                "executor": self.executor.snapshot(),
                "your_last_note": self.tactician_note,
            }
            try:
                result, cost = self.brain.tactics(self.frame_b64(frame), state)
            except (anthropic.APIError, BadResponse) as e:
                self.log(f"tactician error: {e}")
                time.sleep(2)
                continue
            plan = clean_plan(result["plan"])
            accepted = self.executor.set_plan(plan)
            self.tactician_note = result["note"]
            self.last_observation = result["observation"]
            took = time.monotonic() - last
            self.log(f"tactic ({took:.1f}s, ${cost:.4f}, {reason}{', calm' if calm else ''}): "
                     f"{result['observation']} -> {len(plan)} actions{'' if accepted else ' (dropped: reflex active)'}")
            if result["stuck"]:
                self.last_stuck = time.monotonic()
            if result["objective_complete"] or result["stuck"]:
                if self.objective:
                    outcome = "completed" if result["objective_complete"] else "abandoned (stuck)"
                    self.objective_history.append(f"{self.objective['objective']} -> {outcome}")
                self.wake_strategist.set()

    # --- main thread: hotkeys + status line -----------------------------------

    def status_loop(self):
        while self.running:
            time.sleep(10)
            if not self.paused:
                v = self.vitals()
                self.log(f"status: health {v['health']} | hunger {v['hunger']} | "
                         f"{'ACTIVE' if self.is_active() else 'waiting for game focus'}\n  {self.costs.summary()}")

    def run(self):
        self.log("Compiling manifesto...")
        self.rules = self.brain.compile_manifesto(self.statements, self.cfg.get("rules_cache", ".manifesto_rules.json"))
        self.brain.set_manifesto(self.rules)
        self.log(f"Reflexes: flee below {self.rules['flee_below_health']:.0%} health, "
                 f"eat below {self.rules['eat_below_hunger']:.0%} hunger, avoid lava={self.rules['avoid_lava']}")
        self.log("Code: " + " | ".join(self.rules["code_rules"]))
        if self.dry_run:
            self.log("DRY RUN: Claude will plan, but no keys or mouse input will be sent.")

        self.executor.start()
        for fn in (self.reflex_loop, self.strategist_loop, self.tactician_loop, self.status_loop):
            self._spawn(fn)

        winapi.key_pressed(winapi.VK_F8)
        winapi.key_pressed(winapi.VK_F12)
        self.log("Ready. Click into Minecraft, then press F8 to start/pause. F12 quits.")
        try:
            while self.running:
                time.sleep(0.03)
                if winapi.key_pressed(winapi.VK_F12):
                    self.log("F12 pressed, quitting.")
                    break
                if winapi.key_pressed(winapi.VK_F8):
                    self.paused = not self.paused
                    if self.paused:
                        self.executor.set_plan([], source="reflex")
                    self.log("PAUSED (F8 to resume)" if self.paused else "RUNNING (F8 to pause)")
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            self.executor.stop()
            self.executor.join(timeout=2)
            self.executor.release_all()
            self.log(f"Stopped. Total spent this session: ${self.costs.total:.2f} "
                     f"({', '.join(f'{m}: {n} calls' for m, n in self.costs.calls.items())})\n  {self.costs.summary()}")
