"""Everything that talks to Claude: compiling the manifesto into rules,
the slow strategist, the fast tactician, and cost tracking."""

import hashlib
import json
import threading
from pathlib import Path

import anthropic

# $ per million tokens: (input, output). Cache writes bill at 1.25x input, reads at 0.1x.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

CATEGORIES = ("SURVIVAL", "CODE", "AMBITION", "TEMPERAMENT")


class BudgetExceeded(Exception):
    pass


class CostTracker:
    def __init__(self, cap):
        self.cap = cap
        self.total = 0.0
        self.calls = {}
        self.lock = threading.Lock()

    def add(self, model, usage):
        pin, pout = PRICES.get(model, (4.00, 20.00))
        cost = (
            usage.input_tokens * pin
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * pin * 1.25
            + (getattr(usage, "cache_read_input_tokens", 0) or 0) * pin * 0.1
            + usage.output_tokens * pout
        ) / 1_000_000
        with self.lock:
            self.total += cost
            self.calls[model] = self.calls.get(model, 0) + 1
        return cost

    def check(self):
        if self.total >= self.cap:
            raise BudgetExceeded(f"spending cap of ${self.cap:.2f} reached")


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


def _text(response):
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request")
    return next(b.text for b in response.content if b.type == "text")


ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["walk", "look", "mine", "attack", "use", "jump",
                                             "hotbar", "sneak", "wait", "key", "gui_click"]},
        "direction": {"type": "string", "enum": ["forward", "back", "left", "right"]},
        "seconds": {"type": "number"},
        "sprint": {"type": "boolean"},
        "jump": {"type": "boolean"},
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
    },
    "required": ["situation", "objective", "done_when", "hints"],
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
        "temperament_summary": {"type": "string"},
    },
    "required": ["flee_below_health", "eat_below_hunger", "avoid_lava", "fight_or_flight",
                 "survival_summary", "code_rules", "ambition_summary", "temperament_summary"],
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
- The summaries: one or two sentences each, concrete and in Minecraft terms."""

STRATEGIST_SYSTEM = """You are the strategist of an autonomous bot playing Minecraft Bedrock Edition on a friends' Realm, logged in as its owner. You think slowly and decide WHAT the bot should be doing; a fast tactician handles the button presses.

The bot's internal manifesto (in priority order):
1. SURVIVAL (instinct, overrides everything): {survival}
2. CODE (never broken): {code}
3. AMBITION (long-term purpose): {ambition}
4. TEMPERAMENT (style and idle behaviour): {temperament}

Choose ONE concrete objective achievable in roughly 30 seconds to 3 minutes that moves toward the AMBITION, in the style of the TEMPERAMENT, never violating the CODE, and safe per SURVIVAL. Work from what is actually visible in the screenshot and from recent history; the bot is a clumsy player that sees only the screen, so prefer simple, visual objectives ("chop the tree directly ahead until 6 logs", "dig a 1x2 staircase down 10 blocks") over ones that need coordinates or menus. If the previous objective is stuck, pick something different. Respect other players and their builds unless the manifesto says otherwise.

`hints` are practical tips for the tactician (which hotbar slot looks useful, what to look for, what to avoid)."""

TACTICIAN_SYSTEM = """You are the hands of a bot playing Minecraft Bedrock Edition (keyboard + mouse) on a friends' Realm. Every ~1 second you get a fresh screenshot and plan the next 2-5 seconds of actions. Your plan REPLACES whatever is still queued, so always plan from what you see now. Movement should feel continuous: keep moving instead of waiting where sensible.

Manifesto (priority order):
1. SURVIVAL: {survival}
2. CODE (never break these): {code}
3. AMBITION: {ambition}
4. TEMPERAMENT: {temperament}

Actions (JSON objects, executed in order):
- walk: direction forward|back|left|right, seconds, optional sprint, jump (hold jump, for climbing 1-block steps), optional yaw/pitch to turn smoothly WHILE walking.
- look: yaw (degrees, + = right) and pitch (degrees, + = up, - = down). Turning 90 left is yaw -90. Looking straight down at your feet is about pitch -90 from level.
- mine: hold left click on the block under the crosshair until it breaks (seconds = max time, default 4). Aim first with look.
- attack: click times to hit the mob/entity under the crosshair.
- use: hold right click for seconds (place a block ~0.2s, eat ~1.8s, open a door/chest 0.2s).
- jump, hotbar (slot 1-9), sneak (seconds; prevents falling off edges), wait (seconds).
- key: inventory | drop | escape. Only open menus when really needed.
- gui_click: x, y as 0-1 fractions of the screenshot, button left|right; only when a menu is open.

Tips: the crosshair is at the screenshot center; a block outline shows what you're aiming at. Mined blocks drop items you pick up by walking over them. Prefer short plans (3-6 actions). If you see lava, a cliff, or a hostile mob, deal with it first. Read the HUD: hearts bottom-left above the hotbar, hunger drumsticks bottom-right.

Fields: observation = what you see, max 20 words. note = a short memo to your next self (what you're trying, what failed). objective_complete = true once the current objective is achieved. stuck = true if the objective seems impossible from here."""


class Brain:
    def __init__(self, cfg, costs, log=print):
        self.cfg = cfg
        self.costs = costs
        self.log = log
        self.client = anthropic.Anthropic(timeout=60.0, max_retries=2)

    # --- manifesto -> rules (once, cached) --------------------------------

    def compile_manifesto(self, statements, cache_path):
        key = hashlib.sha256((json.dumps(statements, sort_keys=True) + "v1").encode()).hexdigest()
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
        self.costs.add(model, response.usage)
        rules = json.loads(_text(response))
        cache.write_text(json.dumps({"key": key, "rules": rules}, indent=2), encoding="utf-8")
        return rules

    def set_manifesto(self, rules):
        fields = {
            "survival": rules["survival_summary"],
            "code": " ".join(rules["code_rules"]),
            "ambition": rules["ambition_summary"],
            "temperament": rules["temperament_summary"],
        }
        self.strategist_system = STRATEGIST_SYSTEM.format(**fields)
        self.tactician_system = TACTICIAN_SYSTEM.format(**fields)

    # --- strategist -------------------------------------------------------

    def strategize(self, image_b64, state):
        self.costs.check()
        model = self.cfg["strategist_model"]
        response = self.client.beta.messages.create(
            model=model,
            max_tokens=8000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=self.strategist_system,
            output_config={"effort": self.cfg.get("strategist_effort", "medium"),
                           "format": {"type": "json_schema", "schema": STRATEGY_SCHEMA}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": json.dumps(state, indent=1)},
            ]}],
        )
        cost = self.costs.add(model, response.usage)
        return json.loads(_text(response)), cost

    # --- tactician --------------------------------------------------------

    def tactics(self, image_b64, state):
        self.costs.check()
        model = self.cfg["tactician_model"]
        response = self.client.messages.create(
            model=model,
            max_tokens=1200,
            system=[{"type": "text", "text": self.tactician_system, "cache_control": {"type": "ephemeral"}}],
            output_config={"format": {"type": "json_schema", "schema": TACTIC_SCHEMA}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                {"type": "text", "text": json.dumps(state, indent=1)},
            ]}],
        )
        cost = self.costs.add(model, response.usage)
        return json.loads(_text(response)), cost
