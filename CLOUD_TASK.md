# Cloud task: cut the bot's Claude API cost

This file is the brief for a Claude Code cloud session working on this repo. Read it first, then `README.md`.

## What this project is

**Manifesto Bot** plays Minecraft Bedrock Edition on the owner's Windows PC by looking at screenshots and sending real keyboard and mouse input. Its behavior comes from four plain-English statements in `manifesto.txt` (SURVIVAL, CODE, AMBITION, TEMPERAMENT).

It runs four parallel loops (`bot/runner.py`):

| Loop | Model | How often | Job |
|---|---|---|---|
| Strategist | `claude-opus-5-5` (effort `medium`) | every ~30 s, or when woken | Screenshot + recent history → one objective (30 s–3 min of work) |
| Tactician | `claude-haiku-4-5` | ~every 1 s (see the pacing settings below) | Screenshot + state → 2–5 s of actions, replacing the queued tail |
| Executor | local | 100 Hz | Smooth key/mouse input from the action queue |
| Reflexes | local | 10 Hz | Reads hearts, hunger, and lava from pixels; can interrupt anything |

There's also a one-time "compile manifesto" call (the strategist model), cached in `.manifesto_rules.json`.

Key files:
- `bot/brain.py`: prompts, JSON schemas, all Claude calls, `PRICES` table, `CostTracker`
- `bot/runner.py`: the loops and the tactician and strategist pacing logic
- `bot/screen.py`: window capture, `encode_jpeg` (resize + JPEG), HUD pixel reader
- `bot/executor.py`: action primitives (walk, look, mine, attack, use, ...)
- `bot/__main__.py`: CLI (`run`, `run --dry-run`, `plan-once`, `calibrate`, `test-input`)
- `config.toml`: every tunable, including models and pacing

## The goal

**Running the bot costs about $8–12 per hour today. Bring that down as far as possible without making it noticeably worse at playing.** As a rough target, aim for under $3/hour with the default config, and say what it plays like at that price.

Today's settings that drive cost (`config.toml`):

```toml
tactician_model = "claude-haiku-4-5"
strategist_model = "claude-opus-5-5"
strategist_effort = "medium"
strategist_interval_seconds = 30
tactician_min_interval = 0.8
tactician_lookahead_seconds = 1.5
tactician_max_interval = 4.0
screenshot_width = 1024
jpeg_quality = 70
```

### Rough cost model (estimates, verify them)

- **Tactician** is called up to ~3,600 times/hour. Each call sends one ~1024×576 JPEG (roughly 800 image tokens), a cached system prompt, and a JSON state, and gets back up to 1,200 output tokens. At Haiku's $1 in / $5 out per million tokens, **output is likely as expensive as input**: 250 output tokens ≈ $0.00125, about the same as 1,000 input tokens. So the call rate, the image size, and the response length all matter.
- **Strategist** is called ~120 times/hour on Opus ($4 in / $20 out), with `effort: medium` and `max_tokens=8000`. Thinking and output tokens at $20/M can make each call a few cents, so this could be a third or more of the bill.

Check these numbers against the code before relying on them.

## Ideas to evaluate (not a mandate)

Tactician (probably the biggest win):
1. **Skip calls when nothing changed.** If the new frame is nearly identical to the last one sent and the action queue still has plenty of plan left, don't call. A cheap downscaled frame diff with numpy is enough.
2. **Call less often in calm situations.** Lengthen plans (e.g. 4–8 s) when there's no danger, and return to fast pacing when reflexes fire or `stuck` is set.
3. **Shrink the image.** Try 640–768 px width, a lower JPEG quality, or cropping parts of the frame that aren't useful. The HUD is already read locally.
4. **Shrink the output.** Tighter `observation` and `note` limits, a lower `max_tokens`, and compact action JSON.
5. **Keep the cached prefix stable** so prompt caching keeps working, and check that the cache is actually being hit (`cache_read_input_tokens`).

Strategist:
6. Try `claude-sonnet-5-5`, or Opus with `effort: low`, and lower `max_tokens`.
7. Make it **event-driven**: call it when an objective completes or gets stuck, rather than on a fixed 30 s timer, with a long fallback interval.

General:
8. **Per-component cost reporting.** Make the 10 s status line (and `bot.log`) show spend split by tactician and strategist, plus calls per minute and average input and output tokens per call. Then the owner can confirm each saving on a real run.
9. Put new knobs in `config.toml` with comments, keeping the current behavior as an option.

## Constraints for the cloud session

- **The bot cannot run here.** It needs Windows, a visible Minecraft window, and real input (`bot/winapi.py`, `mss`). Don't try to run `python -m bot run`. Changes have to be checked by reading the code, plus small offline checks (e.g. unit-testing a frame-diff function on synthetic numpy arrays, or checking `encode_jpeg` output sizes).
- **There's no API key in the cloud session.** Don't add anything that needs a live Claude call to verify. Token estimates can come from published image-token formulas and the pricing in `bot/brain.py`.
- Keep `PRICES` in `bot/brain.py` correct for any model you switch to.
- Don't change the manifesto system, the CODE rules' meaning, or the safety reflexes. Saving money must not make the bot break other players' builds or ignore danger.
- Keep the existing CLI commands working.

## What to hand back

1. The code and config changes, committed on a branch with a clear summary.
2. A short table in the PR or commit message: each change, its estimated $/hour saving, and any effect on how it plays.
3. A note for the owner on what to test locally: `python -m bot plan-once`, then a few minutes of `python -m bot run` on a single-player world, watching the per-component cost line.
