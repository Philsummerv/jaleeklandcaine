"""Runs the bot: reflexes (local, 10 Hz), tactician (Claude, every 1-8 s),
strategist (Claude, when an objective ends or gets stuck) and the executor, all in parallel."""

import collections
import threading
import time
import traceback
from pathlib import Path

import anthropic

from .brain import BadResponse, Brain, BudgetExceeded, CostTracker, parse_manifesto
from .executor import Executor
from .journal import Journal
from .pacing import (call_reason, frame_change, frame_signature, looks_stuck, settled_damage,
                     should_skip)
from .screen import HudReader, Screen, encode_jpeg

def describe_action(a):
    """A few words per action, so a journal line reads like "walked forward 3s, looked -40"."""
    t = a.get("type", "?")
    if t == "walk":
        return f"walked {a.get('direction', 'forward')} {a.get('seconds', 1):g}s"
    if t == "look":
        return f"looked yaw {a.get('yaw', 0):g} pitch {a.get('pitch', 0):g}"
    if t in ("mine", "use", "wait", "sneak"):
        return f"{t} {a.get('seconds', 1):g}s"
    if t == "attack":
        return f"attacked x{a.get('times', 1)}"
    return t


def describe_action(a):
    """A few words per action, so a journal line reads like "walked forward 3s, looked -40"."""
    t = a.get("type", "?")
    if t == "walk":
        return f"walked {a.get('direction', 'forward')} {a.get('seconds', 1):g}s"
    if t == "look":
        return f"looked yaw {a.get('yaw', 0):g} pitch {a.get('pitch', 0):g}"
    if t in ("mine", "use", "wait", "sneak"):
        return f"{t} {a.get('seconds', 1):g}s"
    if t == "attack":
        return f"attacked x{a.get('times', 1)}"
    return t


MAX_SECONDS = 6.0
# Mining needs its own ceiling. By hand a block takes hardness x 5 seconds: terracotta is
# 6.25s and stone 7.5s, so a 6s cap meant the hold always ended just before the block gave
# and the progress reset. The executor stops as soon as the block breaks anyway.
MINE_MAX_SECONDS = 15.0


def clean_plan(plan):
    """Clamp model output to sane ranges so one weird value can't spin the camera 50 times.
    Returns the plan and a note about anything thrown away, so a silently dropped action
    shows up in the log instead of looking like the bot simply chose to do nothing."""
    out, dropped = [], []
    for a in plan[:10]:
        a = {k: v for k, v in a.items() if v is not None}
        if "seconds" in a:
            cap = MINE_MAX_SECONDS if a.get("type") == "mine" else MAX_SECONDS
            a["seconds"] = max(0.05, min(float(a["seconds"]), cap))
        if "yaw" in a:
            a["yaw"] = max(-180.0, min(float(a["yaw"]), 180.0))
        if "pitch" in a:
            a["pitch"] = max(-90.0, min(float(a["pitch"]), 90.0))
        if "times" in a:
            a["times"] = max(1, min(int(a["times"]), 10))
        if a["type"] == "gui_click" and ("x" not in a or "y" not in a):
            dropped.append("gui_click without x/y")
            continue
        out.append(a)
    return out, dropped


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
        self.pending_window_note = f"Window: {self.screen.describe()}"
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
        self.milestones_done = []
        self.journal = Journal(cfg.get("journal_file", "journal.txt"), cfg.get("journal_lines", 14))
        self.manifesto_path = cfg.get("manifesto_file", "manifesto.txt")
        self.manifesto_stamp = None
        self.techniques_path = cfg.get("techniques_file", "techniques.txt")
        self.techniques_stamp = None

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
        self.journal.add("!", text)
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

    def _save_frame(self, frame, name):
        """Keep the frame that set a reflex off, so the detector can be tuned on real
        pixels rather than on a guess about what the bot was looking at."""
        try:
            from PIL import Image
            Image.fromarray(frame).save(name)
            self.log(f"saved {name} (the frame that triggered this)")
        except Exception as e:
            self.log(f"could not save {name}: {e}")

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
        c = self.cfg
        prev_health = None
        flee_cooldown = lava_cooldown = hunger_nag = 0.0
        low_health_frames = 0
        lava_fires = collections.deque(maxlen=12)
        saved_lava_frame = False
        last_sig = None
        no_hud_frames = 0
        no_hud_cooldown = 0.0
        recent_health = collections.deque(maxlen=3)
        saved_damage_frame = False
        stuck_frames = 0
        stuck_cooldown = 0.0
        stuck_needed = max(int(c.get("stuck_seconds", 1.5) / 0.1), 1)
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
                no_hud_frames = 0
                continue
            if not hud["hud_visible"]:
                # The bars are covered: a menu, the death screen, or a hint popup. The
                # tactician has to click its way out, since nothing here can read it.
                prev_health = hud["health"]
                no_hud_frames += 1
                if no_hud_frames > c.get("no_hud_seconds", 3) / 0.1 and time.monotonic() > no_hud_cooldown:
                    no_hud_cooldown = time.monotonic() + 8
                    self.executor.menu_may_be_open()
                    self.alert("no health or hunger bar on screen for a few seconds: you are on a menu "
                               "or the death screen. Look at the screenshot and click the button you "
                               "can see with gui_click - Respawn if you died, Resume Game if paused")
                continue
            no_hud_frames = 0
            now = time.monotonic()
            rules = self.rules  # re-read each pass: the manifesto can change while running
            health = hud["health"]

            if health is not None:
                recent_health.append(health)
            hit = settled_damage(recent_health, prev_health, c.get("damage_threshold", 0.04),
                                 c.get("damage_spread", 0.12))
            if hit is not None:
                prev_health = hit
                self.last_damage = now
                if not saved_damage_frame and c.get("save_damage_frame", True):
                    saved_damage_frame = True
                    self._save_frame(frame, "damage_trigger.png")
                self.log(f"   (hearts {hud['raw_hearts']} of {self.hud_reader.max_hearts} px, "
                         f"food {hud['raw_food']} of {self.hud_reader.max_food} px)")
                self.alert(f"took damage, health now {hit:.0%}"
                           + ("" if self.last_observation and "zombie" in self.last_observation.lower()
                              else ". If you can't see what hit you it is probably behind you: look yaw 180"))
            elif health is not None and (prev_health is None or health > prev_health):
                prev_health = health      # healing, or the first reading

            # A frame or two below the line is usually a misread, not a wounded player.
            low_health_frames = (low_health_frames + 1 if health is not None
                                 and health < rules["flee_below_health"] else 0)
            if (health is not None and rules["flee_below_health"] > 0
                    and low_health_frames >= c.get("flee_confirm_frames", 3)
                    and now - self.last_damage < 3 and now > flee_cooldown and not self.executor.in_reflex()):
                self.executor.set_plan([
                    {"type": "look", "yaw": 180, "pitch": 0},
                    {"type": "walk", "direction": "forward", "seconds": 2.5, "sprint": True, "jump": True},
                ], source="reflex")
                flee_cooldown = now + 6
                self.alert("REFLEX: low health under attack, fleeing")
                self.wake_strategist.set()  # SURVIVAL may call for a different objective

            if rules["avoid_lava"] and hud["lava"] > c["hud"]["lava_fraction"] and now > lava_cooldown:
                lava_cooldown = now + 2
                lava_fires.append(now)
                recent = sum(1 for t in lava_fires if now - t < c.get("lava_window_seconds", 20))
                if not saved_lava_frame and c.get("save_lava_frame", True):
                    saved_lava_frame = True
                    self._save_frame(frame, "lava_trigger.png")
                if recent <= c.get("lava_max_backoffs", 3):
                    # Sneak: backing away from lava must not walk the bot off a ledge.
                    self.executor.set_plan([{"type": "walk", "direction": "back",
                                             "seconds": 0.6, "sneak": True}], source="reflex")
                    self.alert(f"REFLEX: lava ahead ({hud['lava']:.0%} of view), backed off")
                else:
                    # Orange terrain (badlands especially) reads as lava, and backing up
                    # over and over has walked the bot into holes. Hand it to the tactician.
                    self.alert(f"lava detector has fired {recent}x ({hud['lava']:.0%} of view); not backing "
                               "off again. If this is orange rock and not lava, say so and move on")

            # Walking without the view changing: against a wall, or in a pit.
            if c.get("stuck_detect", True):
                sig = frame_signature(frame)
                change = frame_change(sig, last_sig)
                last_sig = sig
                if looks_stuck(self.executor.snapshot()["doing_now"], change,
                               c.get("stuck_change_threshold", 1.5)):
                    stuck_frames += 1
                else:
                    stuck_frames = 0
                if stuck_frames >= stuck_needed and now > stuck_cooldown:
                    stuck_frames = 0
                    stuck_cooldown = now + c.get("stuck_cooldown_seconds", 6)
                    self.last_stuck = now
                    # Jumping on the spot lands you back where you were. A one-block step is
                    # climbed by moving forward while the jump key is held.
                    self.executor.set_plan([{"type": "walk", "direction": "forward",
                                             "seconds": 0.7, "jump": True}], source="reflex")
                    self.alert("not moving although walking: you are against a wall or in a pit. "
                               "Tried a running jump. If that didn't free you the wall is over one "
                               "block high, so mine your way out or turn and go another way")

            hunger = hud["hunger"]
            if hunger is not None and hunger < rules["eat_below_hunger"] and now > hunger_nag:
                hunger_nag = now + 20
                self.alert(f"hungry ({hunger:.0%}): eat food if you have any (select it in the hotbar, use ~1.8s)")

    # --- manifesto: re-read and recompile when the file changes -------------

    def read_techniques(self):
        try:
            return Path(self.techniques_path).read_text(encoding="utf-8")
        except OSError:
            return ""

    def manifesto_loop(self):
        """Watch both files and apply edits to the running bot. A manifesto change costs one
        compile call (about a cent); identical text is served from the rules cache. A
        techniques change costs nothing - that text is handed to Claude as written."""
        path, tech = Path(self.manifesto_path), Path(self.techniques_path)
        for p, attr in ((path, "manifesto_stamp"), (tech, "techniques_stamp")):
            try:
                setattr(self, attr, p.stat().st_mtime)
            except OSError:
                setattr(self, attr, None)
        while self.running:
            time.sleep(self.cfg.get("manifesto_reload_seconds", 2))
            try:
                tech_stamp = tech.stat().st_mtime
            except OSError:
                tech_stamp = None
            if tech_stamp != self.techniques_stamp:
                self.techniques_stamp = tech_stamp
                self.brain.set_basics(self.read_techniques())
                self.log(f"{tech.name} changed, applied (no recompile needed)")
            try:
                stamp = path.stat().st_mtime
            except OSError:
                continue
            if stamp == self.manifesto_stamp:
                continue
            self.manifesto_stamp = stamp
            try:
                statements = parse_manifesto(path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as e:
                self.log(f"manifesto not usable, keeping the old one: {e}")
                continue
            if statements == self.statements:
                continue
            self.log("manifesto changed, recompiling...")
            try:
                rules = self.brain.compile_manifesto(statements, self.cfg.get("rules_cache", ".manifesto_rules.json"))
            except (anthropic.APIError, BadResponse, BudgetExceeded) as e:
                self.log(f"could not recompile the manifesto, keeping the old one: {e}")
                continue
            self.statements = statements
            self.rules = rules
            self.brain.set_manifesto(rules)
            self.milestones_done = []
            self.objective = None          # the old objective served the old ambition
            self.objective_history.clear()
            self.describe_rules("Manifesto updated.")
            self.wake_strategist.set()

    def describe_rules(self, prefix="Reflexes:"):
        self.log(f"{prefix} flee below {self.rules['flee_below_health']:.0%} health, "
                 f"eat below {self.rules['eat_below_hunger']:.0%} hunger, avoid lava={self.rules['avoid_lava']}")
        self.log("Code: " + " | ".join(self.rules["code_rules"]))
        for i, m in enumerate(self.rules.get("ambition_milestones") or [], 1):
            self.log(f"  milestone {i}: {m}")

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
                "milestones": self.rules.get("ambition_milestones") or [],
                "milestones_done": self.milestones_done,
                "journal": self.journal.recent(),
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
            reached = result.get("milestone")
            if result.get("milestone_complete") and reached and reached not in self.milestones_done:
                self.milestones_done.append(reached)
                self.log(f"MILESTONE REACHED: {reached}")
            self.journal.add("goal", result["objective"])
            self.log(f"STRATEGY (${cost:.3f}): {result['objective']}  |  {result['situation']}"
                     + (f"  [working on: {reached}]" if reached else ""))
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
        now_mining = None
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
                "seconds_on_this_objective": int(time.monotonic() - self.objective_started),
                "journal": self.journal.recent(),
            }
            try:
                result, cost = self.brain.tactics(self.frame_b64(frame), state)
            except (anthropic.APIError, BadResponse) as e:
                self.log(f"tactician error: {e}")
                time.sleep(2)
                continue
            # A mine that changed nothing means the aim was wrong. Say so loudly: the bot
            # reported "crosshair on bark, no outline yet" three calls running and kept
            # swinging at the dirt beside the tree.
            for done in self.executor.snapshot()["recently_done"]:
                if "NOTHING HAPPENED" in done and now_mining != done:
                    now_mining = done
                    self.alert("your last mine did nothing at all - the crosshair was not on a "
                               "block. Before mining again, check for the black outline; if there "
                               "is none, step closer or aim somewhere else. Leaves in front of a "
                               "trunk have to be cleared first")
                    break
            plan, dropped = clean_plan(result["plan"])
            accepted = self.executor.set_plan(plan)
            if dropped:
                self.log("   dropped: " + ", ".join(dropped))
            self.tactician_note = result["note"]
            self.last_observation = result["observation"]
            self.journal.add("saw", f"{result['observation']} -> "
                             + (", ".join(describe_action(a) for a in plan) or "nothing"))
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
        self.log(self.pending_window_note)
        self.log("Compiling manifesto...")
        self.brain.set_basics(self.read_techniques())
        self.rules = self.brain.compile_manifesto(self.statements, self.cfg.get("rules_cache", ".manifesto_rules.json"))
        self.brain.set_manifesto(self.rules)
        self.describe_rules()
        if self.dry_run:
            self.log("DRY RUN: Claude will plan, but no keys or mouse input will be sent.")

        from . import winapi  # Windows-only; imported here so clean_plan can be tested anywhere

        self.journal.reset()
        self.journal.add("start", "run begins")
        self.executor.start()
        for fn in (self.reflex_loop, self.strategist_loop, self.tactician_loop, self.status_loop,
                   self.manifesto_loop):
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
