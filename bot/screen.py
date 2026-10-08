"""Screen capture of the Minecraft window plus cheap pixel-based HUD reading
(health, hunger, lava ahead). Everything here runs locally - no API calls."""

import base64
import collections
import io
import threading

import numpy as np
from PIL import Image

_local = threading.local()


def _winapi():
    from . import winapi  # imported lazily: Windows-only, and the masks below are not
    return winapi


def _sct():
    # mss handles are not thread-safe, so each thread gets its own.
    if not hasattr(_local, "sct"):
        import mss
        _local.sct = mss.MSS()
    return _local.sct


class Screen:
    def __init__(self, window_title):
        self.window_title = window_title
        self.hwnd = None

    def locate(self):
        self.hwnd = _winapi().find_window(self.window_title)
        return self.hwnd

    def describe(self):
        """What window we actually latched onto, so a wrong match is obvious.
        Only an exact title match is certain: the fallback matches any window whose
        title merely starts with window_title, e.g. a browser tab about Minecraft."""
        if not self.hwnd:
            return "no window"
        title = _winapi().window_title(self.hwnd)
        left, top, w, h = _winapi().client_rect(self.hwnd)
        note = "" if title == self.window_title else "  <-- NOT an exact title match, check this is the game"
        return f"'{title}' {w}x{h} at ({left},{top}){note}"


    def rect(self):
        if not self.hwnd:
            self.locate()
        if not self.hwnd:
            raise RuntimeError(f"Can't find a window titled '{self.window_title}'. Is Minecraft open?")
        return _winapi().client_rect(self.hwnd)

    def grab(self, region=None):
        """Grab the client area (or a sub-region given as fractions x0,y0,x1,y1).
        Returns an RGB numpy array."""
        left, top, w, h = self.rect()
        if region:
            x0, y0, x1, y1 = region
            box = {"left": left + int(x0 * w), "top": top + int(y0 * h),
                   "width": max(1, int((x1 - x0) * w)), "height": max(1, int((y1 - y0) * h))}
        else:
            box = {"left": left, "top": top, "width": w, "height": h}
        shot = _sct().grab(box)
        return np.asarray(shot)[:, :, 2::-1]  # BGRA -> RGB

    def focused(self):
        return _winapi().is_foreground(self.hwnd)

    def screen_point(self, fx, fy):
        left, top, w, h = self.rect()
        return left + int(fx * w), top + int(fy * h)


def encode_jpeg(frame, width, quality=70):
    img = Image.fromarray(frame)
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def bar_level(mask, icons=10, min_icon_pixels=20):
    """How full a bar is, read from one frame and nothing else.

    The bar is always `icons` icons wide, so splitting the box into that many columns and
    comparing them to each other gives an absolute reading: the fullest icon in this frame
    is what a full one looks like, whatever the resolution or the GUI scale. No history and
    no calibration, which matters because a bot that starts wounded never sees a full bar -
    one run took 6.5 hearts as its reference and reported 76% health at about 50%.
    """
    width = mask.shape[1]
    edges = [round(i * width / icons) for i in range(icons + 1)]
    counts = [int(mask[:, edges[i]:edges[i + 1]].sum()) for i in range(icons)]
    fullest = max(counts)
    if fullest < min_icon_pixels:
        return None, counts
    return min(sum(counts) / (icons * fullest), 1.0), counts


def _median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _crop(frame, region):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = region
    return frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].astype(np.int16)


def heart_mask(px):
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    return (r > 150) & (g < 70) & (b < 70)


def food_mask(px):
    """Hunger drumsticks, keyed only on their red meat: rgb(208,32,32) and rgb(176,16,16).

    Measured from raw HUD crops. The icon's brown body is deliberately excluded: across
    two frames with the same full hunger bar, the red matched 540 px both times while the
    brown swung from 824 to 270 with the light behind it. Those browns are edge pixels
    blending with the world, so any mask keyed on them tracks the landscape rather than
    the bar. The red does not move.

    This is the same family of threshold as heart_mask, which is fine: the two bars are
    read from separate boxes, so they never compete. Retune from calibration_hud.png.
    """
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    return (r >= 176) & (g <= 72) & (b <= 72)


def lava_mask(px):
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    return (r > 200) & (g > 70) & (g < 190) & (b < 60)


class HudReader:
    """Reads health/hunger as a fraction of the most 'full' bar it has seen.
    It self-calibrates: the first time your bars are full, that becomes 100%."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_hearts = 0
        self.max_food = 0
        self.min_pixels = cfg.get("min_calibration_pixels", 40)
        n = max(int(cfg.get("max_confirm_frames", 3)), 1)
        self._hearts_seen = collections.deque(maxlen=n)
        self._food_seen = collections.deque(maxlen=n)
        m = max(int(cfg.get("smooth_frames", 3)), 1)
        self._hearts_recent = collections.deque(maxlen=m)
        self._food_recent = collections.deque(maxlen=m)
        self.max_fall_per_frame = float(cfg.get("max_fall_per_frame", 0.05))
        self._last_good = None

    @staticmethod
    def _confirm(current_max, seen):
        """Raise the full-bar reference only once a higher reading has held for every
        frame in the window. The bars are drawn straight over the world, so one frame of
        bright terrain behind them would otherwise inflate the reference for the whole
        session and make a full bar read as half empty."""
        if len(seen) < seen.maxlen:
            return current_max
        return max(current_max, min(seen))

    def read(self, frame):
        """Readings are the median of the last few frames. A single frame is not
        trustworthy: anything that briefly hides the HUD - a menu, a screen transition -
        drops both bars to nothing at once, which would otherwise read as a player about
        to die and set the flee reflex off at full health."""
        icons = int(self.cfg.get("bar_icons", 10))
        health_raw, _ = bar_level(heart_mask(_crop(frame, self.cfg["health_region"])), icons)
        hunger_raw, _ = bar_level(food_mask(_crop(frame, self.cfg["hunger_region"])), icons)
        hearts = int(heart_mask(_crop(frame, self.cfg["health_region"])).sum())
        food = int(food_mask(_crop(frame, self.cfg["hunger_region"])).sum())
        lava = float(lava_mask(_crop(frame, self.cfg["lava_region"])).mean())
        # Smooth the fractions, not the pixel counts: one covered frame should not move them.
        self._hearts_recent.append(health_raw if health_raw is not None else 0.0)
        self._food_recent.append(hunger_raw if hunger_raw is not None else 0.0)
        self.max_hearts = max(self.max_hearts, hearts)     # kept for the diagnostic log only
        self.max_food = max(self.max_food, food)
        hearts_s = _median(self._hearts_recent)
        food_s = _median(self._food_recent)
        # Always a number, never None: a bar read as empty is exactly what a covering
        # popup looks like, and the checks below are what decide whether to believe it.
        health, hunger = hearts_s, food_s

        # Hunger is the canary. It drains over minutes and can only rise when the player
        # eats, so a sudden collapse is never real: something is covering the bar. Bedrock's
        # own hint popups ("Scroll or press 2 to hold item") sit right on top of the HUD,
        # which read as 0% hunger and 6% health on a healthy, well-fed player. Health alone
        # cannot be checked this way - a fall really does take most of it at once.
        hud_visible = not (hearts == 0 and food == 0)
        if hud_visible and hunger is not None and self._last_good is not None:
            previous = self._last_good[1]
            if previous is not None and previous - hunger > self.max_fall_per_frame:
                hud_visible = False
        if hud_visible:
            self._last_good = (health, hunger)
        elif self._last_good is not None:
            health, hunger = self._last_good      # keep the last trustworthy reading
        return {"health": health, "hunger": hunger, "lava": lava, "hud_visible": hud_visible,
                "raw_hearts": hearts, "raw_food": food}


def center_patch(screen, size=0.06):
    """Small patch around the crosshair, used to notice when a block breaks."""
    return screen.grab((0.5 - size / 2, 0.5 - size / 2 * 16 / 9, 0.5 + size / 2, 0.5 + size / 2 * 16 / 9)).astype(np.int16)
