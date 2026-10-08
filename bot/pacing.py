"""Decides when the tactician should call Claude. Pure numpy/stdlib (no Windows
or API imports) so it can be unit-tested anywhere."""

import numpy as np

SIG_W, SIG_H = 32, 18


def frame_signature(frame):
    """Tiny grayscale thumbnail (18x32 block means) used to tell whether the view changed."""
    small = np.asarray(frame)[::2, ::2]
    h, w = small.shape[:2]
    bh, bw = h // SIG_H, w // SIG_W
    small = small[:bh * SIG_H, :bw * SIG_W].astype(np.float32)
    if small.ndim == 3:
        small = small.mean(axis=2)
    return small.reshape(SIG_H, bh, SIG_W, bw).mean(axis=(1, 3))


def frame_change(a, b):
    """Mean absolute difference (0-255) between two signatures; inf if either is missing."""
    if a is None or b is None or a.shape != b.shape:
        return float("inf")
    return float(np.abs(a - b).mean())


def call_reason(woken, remaining, since, lookahead, max_interval):
    """Why the tactician wants a new plan right now, or None if it doesn't."""
    if woken:
        return "wake"
    if remaining < lookahead:
        return "low"
    if since > max_interval:
        return "max"
    return None


def should_skip(reason, change, since, threshold, static_interval):
    """Skip a call when the screen looks the same as in the last call:
    - 'max' (plenty of plan still queued): nothing new to react to, keep executing.
    - 'low' (plan running out): wait up to static_interval, since the same picture
      would most likely get the same answer. Wake-ups (alerts, new objective) never skip."""
    if reason == "wake" or change >= threshold:
        return False
    return reason == "max" or since < static_interval


def looks_stuck(doing_now, change, threshold):
    """True when the executor is walking but the view isn't changing: the bot is pushing
    into a wall or treading the floor of a pit. Mining barely moves the view either, so
    only movement counts."""
    return bool(doing_now) and doing_now.startswith("walk") and change < threshold
