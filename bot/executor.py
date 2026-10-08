"""The executor turns action lists into smooth, continuous keyboard/mouse input.

It runs on its own thread at ~100 Hz. New plans replace whatever is left of
the old one, so movement never stops while Claude is thinking. Movement keys
stay held across consecutive walk actions so walking looks continuous."""

import collections
import threading
import time

import numpy as np

from .screen import center_patch

TICK = 0.01


def winapi():
    from . import winapi as mod  # Windows-only; imported lazily so the pure logic tests anywhere
    return mod


DIR_KEYS = {"forward": "w", "back": "s", "left": "a", "right": "d"}
INTERRUPTIBLE = {"walk", "look", "wait", "sneak", "jump"}
MENU_KEYS = {"inventory": "e", "drop": "q", "escape": "esc"}


def estimate_seconds(a):
    t = a.get("type")
    if t in ("walk", "wait", "sneak", "use"):
        return float(a.get("seconds", 1.0))
    if t == "look":
        return 0.1 + (abs(a.get("yaw", 0)) + abs(a.get("pitch", 0))) / 400
    if t == "mine":
        # Usually breaks well before the cap, but assuming 1.5s made the tactician re-plan
        # in the middle of every hand-mined block.
        return min(float(a.get("seconds", 8.0)), 4.0)
    if t == "attack":
        return 0.4 * int(a.get("times", 1))
    return 0.15


def describe(a):
    parts = [a.get("type", "?")]
    for k in ("direction", "seconds", "yaw", "pitch", "slot", "key", "times", "x", "y", "button"):
        if k in a and a[k] is not None:
            v = a[k]
            parts.append(f"{k}={round(v, 2) if isinstance(v, float) else v}")
    for k in ("sprint", "jump"):
        if a.get(k):
            parts.append(k)
    return " ".join(parts)


class Executor(threading.Thread):
    def __init__(self, screen, cfg, is_active, dry_run=False, log=print):
        super().__init__(daemon=True)
        self.screen = screen
        self.cfg = cfg
        self.is_active = is_active  # callable: False when paused or game unfocused
        self.dry_run = dry_run
        self.log = log
        self.ppd = cfg["mouse_pixels_per_degree"]

        self.lock = threading.Lock()
        self.queue = collections.deque()
        self.current = None
        self.current_started = 0.0
        self.abort = threading.Event()
        self.stopped = threading.Event()
        self.history = collections.deque(maxlen=12)
        self.held = set()
        self.buttons = set()
        self.reflex_until = 0.0
        # In Bedrock, escape with no menu open OPENS the pause menu, which freezes the
        # game until something closes it. Only send it when a menu is believed to be open.
        self.menu_open = False
        self.last_action_end = time.monotonic()

    # --- plan management (called from other threads) ----------------------

    def set_plan(self, actions, source="tactician"):
        now = time.monotonic()
        if source != "reflex" and now < self.reflex_until:
            return False
        with self.lock:
            self.queue = collections.deque(actions)
            cur = self.current
            if cur is not None and (source == "reflex" or cur.get("type") in INTERRUPTIBLE):
                self.abort.set()
        if source == "reflex":
            self.reflex_until = now + sum(estimate_seconds(a) for a in actions) + 0.2
        return True

    def remaining_seconds(self):
        with self.lock:
            rest = sum(estimate_seconds(a) for a in self.queue)
            if self.current is not None:
                rest += max(0.0, estimate_seconds(self.current) - (time.monotonic() - self.current_started))
        return rest

    def snapshot(self):
        with self.lock:
            return {
                "doing_now": describe(self.current) if self.current else "idle",
                "still_queued": [describe(a) for a in self.queue],
                "recently_done": list(self.history),
            }

    def menu_may_be_open(self):
        """Told from outside when the HUD has vanished for a while: something is covering
        the screen, so escape is worth allowing again."""
        self.menu_open = True

    def in_reflex(self):
        return time.monotonic() < self.reflex_until

    def stop(self):
        self.stopped.set()
        self.abort.set()

    # --- low-level input with dry-run support -----------------------------

    def _hold(self, keys):
        for k in list(self.held - keys):
            if not self.dry_run:
                winapi().key_up(k)
            self.held.discard(k)
        for k in keys - self.held:
            if not self.dry_run:
                winapi().key_down(k)
            self.held.add(k)

    def _button(self, button, down):
        if down and button not in self.buttons:
            self.buttons.add(button)
            if not self.dry_run:
                winapi().mouse_button(button, True)
        elif not down and button in self.buttons:
            self.buttons.discard(button)
            if not self.dry_run:
                winapi().mouse_button(button, False)

    def release_all(self):
        self._hold(set())
        for b in list(self.buttons):
            self._button(b, False)

    def _move(self, dx, dy):
        if not self.dry_run and (dx or dy):
            winapi().mouse_move_rel(dx, dy)

    def _tap(self, key, hold=0.05):
        if not self.dry_run:
            winapi().tap(key, hold)
        else:
            time.sleep(hold)

    def _ok(self):
        return not self.abort.is_set() and not self.stopped.is_set() and self.is_active()

    def _run_for(self, seconds, yaw=0.0, pitch=0.0, on_tick=None):
        """Wait `seconds`, spreading a camera turn evenly across the time.
        Returns False if interrupted."""
        steps = max(1, int(seconds / TICK))
        px_x, px_y = yaw * self.ppd, -pitch * self.ppd
        sent_x = sent_y = 0.0
        t0 = time.monotonic()
        for i in range(1, steps + 1):
            if not self._ok():
                return False
            tx, ty = px_x * i / steps, px_y * i / steps
            dx, dy = round(tx - sent_x), round(ty - sent_y)
            self._move(dx, dy)
            sent_x += dx
            sent_y += dy
            if on_tick and on_tick():
                return True
            delay = t0 + i * TICK - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return True

    # --- main loop --------------------------------------------------------

    def run(self):
        while not self.stopped.is_set():
            if not self.is_active():
                self.release_all()
                time.sleep(0.05)
                continue
            with self.lock:
                action = self.queue.popleft() if self.queue else None
                self.current = action
                self.current_started = time.monotonic()
                self.abort.clear()
            if action is None:
                # Brief grace period so back-to-back plans don't stutter.
                if self.held and time.monotonic() - self.last_action_end > 0.3:
                    self._hold(set())
                time.sleep(TICK)
                continue
            try:
                result = self._execute(action)
            except Exception as e:  # never let one bad action kill the thread
                result = f"error: {e}"
                self.release_all()
            if result:
                self.history.append(f"{describe(action)} -> {result}")
            with self.lock:
                self.current = None
            self.last_action_end = time.monotonic()
        self.release_all()

    def _execute(self, a):
        t = a.get("type")
        if t != "walk":
            self._hold(set())

        if t == "walk":
            keys = {DIR_KEYS.get(a.get("direction", "forward"), "w")}
            if a.get("sprint") and a.get("direction", "forward") == "forward":
                keys.add("ctrl")
            if a.get("jump"):
                keys.add("space")
            if a.get("sneak"):
                keys.add("shift")  # sneaking stops you walking off a ledge
            self._hold(keys)
            done = self._run_for(float(a.get("seconds", 1.0)), a.get("yaw") or 0.0, a.get("pitch") or 0.0)
            if a.get("jump"):
                self._hold(keys - {"space"})
            return "ok" if done else "interrupted"

        if t == "look":
            yaw, pitch = a.get("yaw") or 0.0, a.get("pitch") or 0.0
            done = self._run_for(estimate_seconds(a), yaw, pitch)
            return "ok" if done else "interrupted"

        if t == "mine":
            return self._mine(float(a.get("seconds", 8.0)))

        if t == "attack":
            n = 0
            for _ in range(max(1, int(a.get("times", 1)))):
                if not self._ok():
                    break
                self._button("left", True)
                time.sleep(0.05)
                self._button("left", False)
                n += 1
                self._run_for(0.35)
            return f"swung {n}x"

        if t == "use":
            self._button("right", True)
            try:
                done = self._run_for(float(a.get("seconds", 0.2)))
            finally:
                self._button("right", False)
            return "ok" if done else "interrupted"

        if t == "jump":
            self._tap("space", 0.1)
            return "ok"

        if t == "hotbar":
            slot = int(a.get("slot", 1))
            if 1 <= slot <= 9:
                self._tap(str(slot))
            return "ok"

        if t == "sneak":
            self._hold({"shift"})
            done = self._run_for(float(a.get("seconds", 1.0)))
            self._hold(set())
            return "ok" if done else "interrupted"

        if t == "wait":
            done = self._run_for(float(a.get("seconds", 0.5)))
            return "ok" if done else "interrupted"

        if t == "key":
            name = a.get("key")
            if name == "escape" and not self.menu_open:
                return "escape ignored: no menu was open (it would have opened the pause menu)"
            key = MENU_KEYS.get(name)
            if key:
                self._tap(key)
                time.sleep(0.2)
                if name == "inventory":
                    self.menu_open = not self.menu_open
                elif name == "escape":
                    self.menu_open = False
            return "ok" if key else "unknown key"

        if t == "gui_click":
            x, y = self.screen.screen_point(float(a.get("x", 0.5)), float(a.get("y", 0.5)))
            button = a.get("button", "left")
            if not self.dry_run:
                winapi().mouse_move_abs(x, y)
                time.sleep(0.08)
            self._button(button, True)
            time.sleep(0.05)
            self._button(button, False)
            time.sleep(0.1)
            return "ok"

        return "unknown action"

    def _mine(self, max_seconds):
        """Hold left click until the block under the crosshair changes a lot
        (it broke) or we time out."""
        threshold = self.cfg["mine_break_threshold"]
        self._button("left", True)
        try:
            if not self._run_for(0.2):
                return "interrupted"
            base = center_patch(self.screen)
            t0 = time.monotonic()
            while time.monotonic() - t0 < max_seconds:
                if not self._ok():
                    return "interrupted"
                diff = float(np.abs(center_patch(self.screen) - base).mean())
                if diff > threshold:
                    self._run_for(0.05)
                    return f"block broke after {time.monotonic() - t0 + 0.2:.1f}s"
                time.sleep(0.05)
            return f"nothing broke in {max_seconds:.0f}s (wrong aim, too hard, or air?)"
        finally:
            self._button("left", False)
