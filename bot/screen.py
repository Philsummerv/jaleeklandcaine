"""Screen capture of the Minecraft window plus cheap pixel-based HUD reading
(health, hunger, lava ahead). Everything here runs locally - no API calls."""

import base64
import io
import threading

import mss
import numpy as np
from PIL import Image

from . import winapi

_local = threading.local()


def _sct():
    # mss handles are not thread-safe, so each thread gets its own.
    if not hasattr(_local, "sct"):
        _local.sct = mss.MSS()
    return _local.sct


class Screen:
    def __init__(self, window_title):
        self.window_title = window_title
        self.hwnd = None

    def locate(self):
        self.hwnd = winapi.find_window(self.window_title)
        return self.hwnd

    def describe(self):
        """What window we actually latched onto, so a wrong match is obvious.
        Only an exact title match is certain: the fallback matches any window whose
        title merely starts with window_title, e.g. a browser tab about Minecraft."""
        if not self.hwnd:
            return "no window"
        title = winapi.window_title(self.hwnd)
        left, top, w, h = winapi.client_rect(self.hwnd)
        note = "" if title == self.window_title else "  <-- NOT an exact title match, check this is the game"
        return f"'{title}' {w}x{h} at ({left},{top}){note}"


    def rect(self):
        if not self.hwnd:
            self.locate()
        if not self.hwnd:
            raise RuntimeError(f"Can't find a window titled '{self.window_title}'. Is Minecraft open?")
        return winapi.client_rect(self.hwnd)

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
        return winapi.is_foreground(self.hwnd)

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


def _crop(frame, region):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = region
    return frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].astype(np.int16)


def heart_mask(px):
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    return (r > 150) & (g < 70) & (b < 70)


def food_mask(px):
    # Drumstick meat: orange/brown.
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    return (r > 140) & (g > 60) & (g < 150) & (b < 70) & (r - g > 40)


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

    def read(self, frame):
        hearts = int(heart_mask(_crop(frame, self.cfg["health_region"])).sum())
        food = int(food_mask(_crop(frame, self.cfg["hunger_region"])).sum())
        lava = float(lava_mask(_crop(frame, self.cfg["lava_region"])).mean())
        self.max_hearts = max(self.max_hearts, hearts)
        self.max_food = max(self.max_food, food)
        health = hearts / self.max_hearts if self.max_hearts >= self.min_pixels else None
        hunger = food / self.max_food if self.max_food >= self.min_pixels else None
        return {"health": health, "hunger": hunger, "lava": lava,
                "raw_hearts": hearts, "raw_food": food}


def center_patch(screen, size=0.06):
    """Small patch around the crosshair, used to notice when a block breaks."""
    return screen.grab((0.5 - size / 2, 0.5 - size / 2 * 16 / 9, 0.5 + size / 2, 0.5 + size / 2 * 16 / 9)).astype(np.int16)
