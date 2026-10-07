# Manifesto Bot

A bot that plays Minecraft Bedrock Edition on your PC, as you, guided by four statements in `manifesto.txt`:

| Category | Job |
|---|---|
| **SURVIVAL** | Instinct. Overrides everything; becomes fast local reflexes (flee, eat, avoid lava). |
| **CODE** | Ethics. Hard rules it never breaks. |
| **AMBITION** | Purpose. The long-term goal it works toward. |
| **TEMPERAMENT** | Personality. Its style, and what it does when idle. |

## How it works

```
 Strategist (Opus, on events)    --- picks an objective when the last one ends or gets stuck (fallback every 2 min)
        |
 Tactician  (Haiku, every 1-8s)  --- screenshot -> 2-8 s of actions; new plans replace the old tail
        |
 Executor   (local, 100 Hz)      --- smooth key/mouse input; keeps walking while Claude thinks
        ^
 Reflexes   (local, 10 Hz)       --- reads hearts/hunger/lava from pixels and can interrupt anything
```

## Setup

1. `pip install -r requirements.txt`
2. Create a file named `.env` in this folder containing `ANTHROPIC_API_KEY=sk-ant-...`.
3. In Minecraft settings, turn on **Auto-Jump**, and use windowed or borderless mode at 1920×1080 if possible.
4. Edit `manifesto.txt`.

## First run: calibrate on a single-player world

1. `python -m bot test-input`: click into the game during the countdown. The bot walks, jumps, turns 360° and looks down/up. Adjust `mouse_pixels_per_degree` in `config.toml` until the 360° turn ends where it started.
2. `python -m bot calibrate`: run it with full health and hunger, then open `calibration.png`. The green box should cover your hearts and the blue box your hunger bar, with matched pixels highlighted. If they don't, adjust the `[hud]` regions.
3. `python -m bot plan-once`: one strategist and tactician call against the current screen, printed, with no input sent. This costs about a cent and shows what Claude thinks.
4. `python -m bot run --dry-run`: the full loop, but no keys are pressed.
5. `python -m bot run`: click into Minecraft, then press **F8** to start or pause and **F12** to quit.

The bot automatically pauses (and releases all keys) whenever Minecraft isn't the focused window. When you're happy with it on a single-player world, join the Realm and run it the same way.

## Cost

Every Claude call is priced from its real token usage. A status line every 10 s (also in `bot.log`) shows the total, a $/hour projection, and for the tactician and strategist separately: spend, calls per minute, skipped calls, average input tokens per call with the share read from cache, and average output tokens. The bot stops itself at `max_dollars_per_session` (default $5).

Expect roughly **$1/hour** with the default config (it was $8–12/hour before the cost work). Most of the saving comes from:

- the tactician on `claude-haiku-5-5` ($0.10/$0.50 per million tokens), with its system prompt read from cache;
- adaptive pacing: longer plans and fewer calls when nothing is happening, plus skipping calls when the screen hasn't changed;
- a 768 px screenshot (~440 image tokens instead of ~790);
- the strategist at `effort = "low"`, called when an objective ends instead of every 30 s.

Every one of these is a setting in `config.toml`, with the old value noted next to it. `python -m unittest discover tests` runs offline checks of the pacing and cost logic.

## Files

- `bot/executor.py`: action primitives (walk, look, mine, attack, use, ...) and smooth input
- `bot/screen.py`: window capture and the pixel HUD reader
- `bot/brain.py`: prompts, schemas, Claude calls and cost tracking
- `bot/runner.py`: the four parallel loops
- `bot.log`: everything the bot saw and decided
