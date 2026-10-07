"""Command line entry point.

    python -m bot run [--dry-run]   play (F8 start/pause, F12 quit)
    python -m bot plan-once         one strategist + tactician call, prints the plan, no input
    python -m bot calibrate         save calibration.png showing what the HUD reader sees
    python -m bot test-input        walk/turn/jump test to check controls and mouse sensitivity
"""

import argparse
import json
import os
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    """Read KEY=value lines from .env into the environment (without overriding)."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load():
    load_env()
    cfg =tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))
    from .brain import parse_manifesto
    statements = parse_manifesto((ROOT / "manifesto.txt").read_text(encoding="utf-8"))
    return cfg, statements


def countdown(screen, seconds=5):
    print(f"Click into the Minecraft window. Starting in {seconds}s...")
    for i in range(seconds, 0, -1):
        print(f"  {i}", flush=True)
        time.sleep(1)
    if not screen.focused():
        sys.exit("Minecraft isn't the focused window - aborting so we don't type into something else.")


def cmd_run(args):
    from .runner import Bot
    cfg, statements = load()
    Bot(cfg, statements, dry_run=args.dry_run).run()


def cmd_plan_once(args):
    from .brain import Brain, CostTracker
    from .screen import Screen, encode_jpeg
    cfg, statements = load()
    screen = Screen(cfg["window_title"])
    if not screen.locate():
        sys.exit("Minecraft window not found.")
    costs = CostTracker(cfg["max_dollars_per_session"])
    brain = Brain(cfg, costs)
    rules = brain.compile_manifesto(statements, ROOT / cfg.get("rules_cache", ".manifesto_rules.json"))
    print("Compiled rules:\n" + json.dumps(rules, indent=2))
    brain.set_manifesto(rules)
    img = encode_jpeg(screen.grab(), cfg["screenshot_width"], cfg.get("jpeg_quality", 70))
    strategy, c1 = brain.strategize(img, {"current_objective": None, "previous_objectives": []})
    print(f"\nStrategist (${c1:.4f}):\n" + json.dumps(strategy, indent=2))
    t0 = time.monotonic()
    tactic, c2 = brain.tactics(img, {"objective": strategy["objective"], "done_when": strategy["done_when"],
                                     "strategist_hints": strategy["hints"], "plan_seconds": "2-5",
                                     "executor": {"doing_now": "idle"}})
    print(f"\nTactician ({time.monotonic() - t0:.1f}s, ${c2:.4f}):\n" + json.dumps(tactic, indent=2))
    print(f"\nTotal: ${costs.total:.4f}\n  {costs.summary()}")


def cmd_calibrate(args):
    from PIL import Image, ImageDraw
    from .screen import HudReader, Screen, food_mask, heart_mask, lava_mask
    cfg, _ = load()
    screen = Screen(cfg["window_title"])
    if not screen.locate():
        sys.exit("Minecraft window not found.")
    frame = screen.grab()
    h, w = frame.shape[:2]
    reading = HudReader({**cfg["hud"], "min_calibration_pixels": 1}).read(frame)
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    for name, mask, color in (("health_region", heart_mask, (0, 255, 0)),
                              ("hunger_region", food_mask, (0, 200, 255)),
                              ("lava_region", lava_mask, (255, 0, 255))):
        x0, y0, x1, y1 = cfg["hud"][name]
        box = (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 4, box[1] + 4), name, fill=color)
        # Highlight matched pixels so you can see what's being counted.
        crop = frame[box[1]:box[3], box[0]:box[2]]
        ys, xs = mask(crop.astype("int16")).nonzero()
        for y, x in zip(ys[::3], xs[::3]):
            img.putpixel((box[0] + int(x), box[1] + int(y)), color)
    out = ROOT / "calibration.png"
    img.save(out)
    print(f"Window: {w}x{h}. Heart pixels: {reading['raw_hearts']}, food pixels: {reading['raw_food']}, "
          f"lava fraction: {reading['lava']:.3f}")
    print(f"Saved {out}. The green box should contain your hearts and the blue box your hunger bar;"
          " adjust [hud] regions in config.toml if not.")


def cmd_test_input(args):
    from . import winapi
    from .screen import Screen
    cfg, _ = load()
    screen = Screen(cfg["window_title"])
    if not screen.locate():
        sys.exit("Minecraft window not found.")
    countdown(screen)
    ppd = cfg["mouse_pixels_per_degree"]
    print("Walking forward 1s...")
    winapi.key_down("w"); time.sleep(1); winapi.key_up("w")
    print("Jumping...")
    winapi.tap("space", 0.1); time.sleep(0.6)
    print("Turning a full 360 degrees - you should end up facing the same way you started.")
    for _ in range(100):
        winapi.mouse_move_rel(360 * ppd / 100, 0)
        time.sleep(0.01)
    time.sleep(0.5)
    print("Looking down 45 degrees then back up...")
    for dy in (45, -45):
        for _ in range(30):
            winapi.mouse_move_rel(0, dy * ppd / 30)
            time.sleep(0.01)
        time.sleep(0.3)
    print("Done. If the 360 turn over- or under-shot, change mouse_pixels_per_degree in config.toml\n"
          "(overshot -> lower it, undershot -> raise it).")


def main():
    p = argparse.ArgumentParser(prog="python -m bot", description="Manifesto-driven Minecraft Bedrock bot")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="play")
    r.add_argument("--dry-run", action="store_true", help="plan with Claude but send no input")
    sub.add_parser("plan-once", help="one planning round, printed, no input")
    sub.add_parser("calibrate", help="check the HUD reader regions")
    sub.add_parser("test-input", help="check controls and mouse sensitivity")
    args = p.parse_args()
    {"run": cmd_run, "plan-once": cmd_plan_once, "calibrate": cmd_calibrate,
     "test-input": cmd_test_input}[args.cmd](args)


if __name__ == "__main__":
    main()
