"""Everything that talks to Claude: compiling the manifesto into rules,
the slow strategist, the fast tactician, and cost tracking."""

import hashlib
import json
import threading
import time
from pathlib import Path

import anthropic

# $ per million tokens: (input, output). Cache writes bill at 1.25x input, reads at 0.1x.
# Haiku 5.5 is $0.10/$0.50 for prompts up to 100K tokens (ours are ~2K).
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

CATEGORIES = ("SURVIVAL", "CODE", "AMBITION", "TEMPERAMENT")


class BudgetExceeded(Exception):
    pass


class BadResponse(Exception):
    """A response the bot can't use (refusal, cut off, invalid JSON). Recoverable: try again."""


class CostTracker:
    """Prices every call from its real usage and keeps per-component totals
    (tactician / strategist / compile) for the status line."""

    FIELDS = ("calls", "skipped", "cost", "input", "cache_write", "cache_read", "output")

    def __init__(self, cap):
        self.cap = cap
        self.total = 0.0
        self.calls = {}
        self.parts = {}
        self.started = None
        self.lock = threading.Lock()

    def _part(self, component):
        return self.parts.setdefault(component, dict.fromkeys(self.FIELDS, 0))

    def add(self, model, usage, component="other"):
        pin, pout = PRICES.get(model, (4.00, 20.00))
        fresh = usage.input_tokens
        write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cost = (fresh * pin + write * pin * 1.25 + read * pin * 0.1 + usage.output_tokens * pout) / 1_000_000
        with self.lock:
            if self.started is None:
                self.started = time.monotonic()
            self.total += cost
            self.calls[model] = self.calls.get(model, 0) + 1
            p = self._part(component)
            p["calls"] += 1
            p["cost"] += cost
            p["input"] += fresh
            p["cache_write"] += write
            p["cache_read"] += read
            p["output"] += usage.output_tokens
        return cost

    def skip(self, component):
        """Count a call that was skipped because nothing on screen changed."""
        with self.lock:
            self._part(component)["skipped"] += 1

    def check(self):
        if self.total >= self.cap:
            raise BudgetExceeded(f"spending cap of ${self.cap:.2f} reached")

    def summary(self):
        """One line per component: spend, $/hour, calls/min, and average tokens per call."""
        with self.lock:
            minutes = max((time.monotonic() - self.started) / 60, 1 / 60) if self.started else None
            lines = [f"spent ${self.total:.2f} of ${self.cap:.2f}"
                     + (f", ~${self.total / minutes * 60:.2f}/hour" if minutes else "")]
            for name, p in self.parts.items():
                n = max(p["calls"], 1)
                cached = p["cache_read"] / max(p["input"] + p["cache_write"] + p["cache_read"], 1)
                lines.append(
                    f"{name}: ${p['cost']:.3f} | {p['calls'] / minutes:.1f} calls/min"
                    + (f" (+{p['skipped'] / minutes:.1f} skipped)" if p["skipped"] else "")
                    + f" | avg in {(p['input'] + p['cache_write'] + p['cache_read']) / n:.0f}"
                    f" ({cached:.0%} cached), out {p['output'] / n:.0f}"
                    if minutes else f"{name}: no calls yet")
            return "\n  ".join(lines)


def parse_manifesto(text):
    """Pull the four statements out of manifesto.txt (lines like 'SURVIVAL: ...')."""
    found = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        head, body = line.split(":", 1)
        head = head.strip().upper()
        if head in CATEGORIES and body.strip():
            found[head] = body.strip()
    missing = [c for c in CATEGORIES if c not in found]
    if missing:
        raise ValueError(f"manifesto.txt is missing: {', '.join(missing)}")
    return found


def _json(response):
    if response.stop_reason == "refusal":
        raise BadResponse("Claude declined this request")
    if response.stop_reason == "max_tokens":
        raise BadResponse("response was cut off at max_tokens")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise BadResponse("response had no text")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise BadResponse(f"invalid JSON: {e}") from None


def _compact(state):
    return json.dumps(state, separators=(",", ":"), ensure_ascii=False)


ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["walk", "look", "mine", "attack", "use", "jump",
                                             "hotbar", "sneak", "wait", "key", "gui_click"]},
        "direction": {"type": "string", "enum": ["forward", "back", "left", "right"]},
        "seconds": {"type": "number"},
        "sprint": {"type": "boolean"},
        "jump": {"type": "boolean"},
        "sneak": {"type": "boolean"},
        "yaw": {"type": "number"},
        "pitch": {"type": "number"},
        "slot": {"type": "integer"},
        "times": {"type": "integer"},
        "key": {"type": "string", "enum": ["inventory", "drop", "escape"]},
        "x": {"type": "number"},
        "y": {"type": "number"},
        "button": {"type": "string", "enum": ["left", "right"]},
    },
    "required": ["type"],
    "additionalProperties": False,
}

TACTIC_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "plan": {"type": "array", "items": ACTION_SCHEMA},
        "note": {"type": "string"},
        "objective_complete": {"type": "boolean"},
        "stuck": {"type": "boolean"},
    },
    "required": ["observation", "plan", "note", "objective_complete", "stuck"],
    "additionalProperties": False,
}

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "situation": {"type": "string"},
        "objective": {"type": "string"},
        "done_when": {"type": "string"},
        "hints": {"type": "string"},
        "milestone": {"type": "string"},
        "milestone_complete": {"type": "boolean"},
    },
    "required": ["situation", "objective", "done_when", "hints", "milestone", "milestone_complete"],
    "additionalProperties": False,
}

RULES_SCHEMA = {
    "type": "object",
    "properties": {
        "flee_below_health": {"type": "number"},
        "eat_below_hunger": {"type": "number"},
        "avoid_lava": {"type": "boolean"},
        "fight_or_flight": {"type": "string", "enum": ["fight", "flee", "depends"]},
        "survival_summary": {"type": "string"},
        "code_rules": {"type": "array", "items": {"type": "string"}},
        "ambition_summary": {"type": "string"},
        "ambition_milestones": {"type": "array", "items": {"type": "string"}},
        "temperament_summary": {"type": "string"},
    },
    "required": ["flee_below_health", "eat_below_hunger", "avoid_lava", "fight_or_flight",
                 "survival_summary", "code_rules", "ambition_summary", "ambition_milestones",
                 "temperament_summary"],
    "additionalProperties": False,
}

COMPILE_PROMPT = """You are configuring an autonomous Minecraft (Bedrock Edition) bot from its owner's four-part manifesto.

The manifesto has four categories with different jobs:
- SURVIVAL: instinct. Overrides everything when triggered. Becomes fast local reflexes.
- CODE: ethics. Hard vetoes the bot must never break.
- AMBITION: purpose. The long-term goal it works toward when safe.
- TEMPERAMENT: personality. How it acts, its style, what it does when idle and how it breaks ties.

Manifesto:
SURVIVAL: {SURVIVAL}
CODE: {CODE}
AMBITION: {AMBITION}
TEMPERAMENT: {TEMPERAMENT}

Translate this into configuration:
- flee_below_health / eat_below_hunger: fractions 0.0-1.0 for the local reflexes. Infer sensible values from SURVIVAL (and TEMPERAMENT if relevant); default 0.4 / 0.5 if unspecified. Use 0 to disable fleeing if the manifesto says never to retreat.
- avoid_lava: true unless the manifesto clearly says otherwise.
- fight_or_flight: the default reaction to hostile mobs.
- code_rules: rewrite CODE as 1-5 short, concrete, checkable rules ("Never attack cows, pigs, sheep, chickens or other passive mobs").
- The summaries: one or two sentences each, concrete and in Minecraft terms.
- ambition_milestones: 5-8 milestones in order, each one concrete enough to look at the screen and say whether it is done ("holding a stone pickaxe", "32 logs in the inventory").

The owner may have written an AMBITION that is only a first step, or one that would be finished in ten minutes. Work out what it is FOR, and carry the ladder well past where their words stop: the early milestones are what they asked for, the later ones are what someone who wanted that would want next, in the style of the TEMPERAMENT and never breaking the CODE. Something asking only for wood should end up somewhere a player with plenty of wood would go. Keep every milestone achievable by a clumsy player who can only see the screen, and order them so each one makes the next easier."""

STRATEGIST_SYSTEM = """You are the strategist of an autonomous bot playing Minecraft Bedrock Edition on a friends' Realm, logged in as its owner. You think slowly and decide WHAT the bot should be doing; a fast tactician handles the button presses.

The bot's internal manifesto (in priority order):
1. SURVIVAL (instinct, overrides everything): {survival}
2. CODE (never broken): {code}
3. AMBITION (long-term purpose): {ambition}
4. TEMPERAMENT (style and idle behaviour): {temperament}

The ladder of milestones toward the AMBITION, in order:
{milestones}

Work the earliest milestone that is not done yet, and say which one in `milestone`. Set `milestone_complete` only when the screenshot or the history shows that milestone is actually finished, not merely started. If every milestone is done, carry the AMBITION onward yourself: name the next worthwhile goal beyond the list and work toward that.

Choose ONE concrete objective achievable in roughly 30 seconds to 3 minutes that moves toward the AMBITION, in the style of the TEMPERAMENT, never violating the CODE, and safe per SURVIVAL. Work from what is actually visible in the screenshot and from recent history; the bot is a clumsy player that sees only the screen, so prefer simple, visual objectives ("chop the tree directly ahead until 6 logs", "dig a 1x2 staircase down 10 blocks") over ones that need coordinates or menus. If the previous objective is stuck, pick something different. Respect other players and their builds unless the manifesto says otherwise.

`hints` are practical tips for the tactician (which hotbar slot looks useful, what to look for, what to avoid)."""

TACTICIAN_SYSTEM = """You are the hands of a bot playing Minecraft Bedrock Edition (keyboard + mouse) on a friends' Realm. You get a fresh screenshot whenever the current plan is running out or something happens, and plan the next few seconds of actions: `plan_seconds` in the state says how long the plan should last (short when there is danger, longer when things are calm). Your plan REPLACES whatever is still queued, so always plan from what you see now. Movement should feel continuous: keep moving instead of waiting where sensible.

Manifesto (priority order):
1. SURVIVAL: {survival}
2. CODE (never break these): {code}
3. AMBITION: {ambition}
4. TEMPERAMENT: {temperament}

Actions (JSON objects, executed in order):
- walk: direction forward|back|left|right, seconds, optional sprint, jump (hold jump, for climbing 1-block steps), sneak (move slowly without falling off edges - use it near any drop), optional yaw/pitch to turn smoothly WHILE walking.
- look: yaw (degrees, + = right) and pitch (degrees, + = up, - = down). Turning 90 left is yaw -90. Looking straight down at your feet is about pitch -90 from level.
- mine: hold left click on the block under the crosshair until it breaks (seconds = max time, default 8). Aim first with look. Bare-handed a block takes hardness x 5 seconds: leaves and grass are instant, dirt and sand about 2s, wood about 3s, and stone, terracotta and ore about 7s - so pass seconds: 12 for anything stony while you have no pickaxe, or the hold ends just before the block gives and all the progress is lost. Stone, terracotta and ore drop NOTHING without a pickaxe in hand, so mining them bare-handed only clears a path; wood and dirt always drop.
- attack: click times to hit the mob/entity under the crosshair.
- use: hold right click for seconds (place a block ~0.2s, eat ~1.8s, open a door/chest 0.2s).
- jump, hotbar (slot 1-9), sneak (seconds; prevents falling off edges), wait (seconds).
- key: inventory | drop | escape. Only open menus when really needed.
- gui_click: x, y as 0-1 fractions of the screenshot, button left|right; only when a menu is open.

Crafting (Bedrock): press key inventory to open it, then use gui_click. Bedrock lists the recipes you can currently make down the side of the crafting tab, so you usually just click the recipe and then the result - you do not have to arrange items in a grid. Planks, sticks and a crafting table can be made from your own inventory anywhere. A pickaxe, axe or sword needs a crafting table placed in the world: select it in the hotbar, aim at flat ground, use to place it, then look at it and use to open it. Always press key escape to close a menu before you move, or your movement keys go into the menu.

Keep the horizon in view while travelling, pitch roughly level: you cannot spot trees, mobs or cliffs while staring at your feet. Look down only to mine or place a block right at your feet, then look back up. If the whole screen is one texture you are probably facing into a wall or a hole, so back out and raise your view.

Tips: the crosshair is at the screenshot center; a block outline shows what you're aiming at. Mined blocks drop items you pick up by walking over them. Prefer 3-8 actions. If you see lava, a cliff, or a hostile mob, deal with it first. Read the HUD: hearts bottom-left above the hotbar, hunger drumsticks bottom-right.

Fields: observation = what you see, max 12 words. note = a memo to your next self (what you're trying, what failed), max 15 words. In actions, only include the fields that action uses. objective_complete = true once the current objective is achieved. stuck = true if the objective seems impossible from here."""


class Brain:
    def __init__(self, cfg, costs, log=print):
        self.cfg = cfg
        self.costs = costs
        self.log = log
        self.client = anthropic.Anthropic(timeout=60.0, max_retries=2)

    # --- manifesto -> rules (once, cached) --------------------------------

    def compile_manifesto(self, statements, cache_path):
        key = hashlib.sha256((json.dumps(statements, sort_keys=True) + "v2").encode()).hexdigest()
        cache = Path(cache_path)
        if cache.exists():
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("key") == key:
                return data["rules"]
        self.log("Compiling manifesto with Claude (one-time)...")
        model = self.cfg["strategist_model"]
        response = self.client.beta.messages.create(
            model=model,
            max_tokens=8000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium",
                           "format": {"type": "json_schema", "schema": RULES_SCHEMA}},
            messages=[{"role": "user", "content": COMPILE_PROMPT.format(**statements)}],
        )
        self.costs.add(model, response.usage, "compile")
        rules = _json(response)
        cache.write_text(json.dumps({"key": key, "rules": rules}, indent=2), encoding="utf-8")
        return rules

    def set_manifesto(self, rules):
        fields = {
            "survival": rules["survival_summary"],
            "code": " ".join(rules["code_rules"]),
            "ambition": rules["ambition_summary"],
            "temperament": rules["temperament_summary"],
        }
        milestones = rules.get("ambition_milestones") or ["(none worked out)"]
        self.strategist_system = STRATEGIST_SYSTEM.format(
            milestones="\n".join(f"{i}. {m}" for i, m in enumerate(milestones, 1)), **fields)
        self.tactician_system = TACTICIAN_SYSTEM.format(**fields)

    # --- strategist -------------------------------------------------------

    def strategize(self, image_b64, state):
        self.costs.check()
        model = self.cfg["strategist_model"]
        response = self.client.beta.messages.create(
            model=model,
            max_tokens=self.cfg.get("strategist_max_tokens", 8000),
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=[{"type": "text", "text": self.strategist_system, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": self.cfg.get("strategist_effort", "medium"),
                           "format": {"type": "json_schema", "schema": STRATEGY_SCHEMA}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": _compact(state)},
            ]}],
        )
        cost = self.costs.add(model, response.usage, "strategist")
        return _json(response), cost

    # --- tactician --------------------------------------------------------

    def tactics(self, image_b64, state):
        self.costs.check()
        model = self.cfg["tactician_model"]
        output_config = {"format": {"type": "json_schema", "schema": TACTIC_SCHEMA}}
        if self.cfg.get("tactician_effort"):  # not supported by claude-haiku-4-5
            output_config["effort"] = self.cfg["tactician_effort"]
        response = self.client.messages.create(
            model=model,
            max_tokens=self.cfg.get("tactician_max_tokens", 1200),
            thinking={"type": self.cfg.get("tactician_thinking", "disabled")},
            # The system prompt is identical on every call, so it's read from cache at 0.1x.
            system=[{"type": "text", "text": self.tactician_system, "cache_control": {"type": "ephemeral"}}],
            output_config=output_config,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": _compact(state)},
            ]}],
        )
        cost = self.costs.add(model, response.usage, "tactician")
        return _json(response), cost
