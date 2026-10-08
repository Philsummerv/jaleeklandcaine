"""A small window for editing the four manifesto statements while the bot is running.

Apply writes manifesto.txt; the running bot notices the file changed, recompiles its
rules and milestones, and picks a fresh objective. Tkinter is in the standard library,
so this adds no dependencies.

Clicking into this window takes focus away from Minecraft, which pauses the bot. It
starts moving again when you click back into the game.
"""

import json
import tkinter as tk
from tkinter import ttk

from .brain import CATEGORIES, parse_manifesto

BLURB = {
    "SURVIVAL": "Instinct. Overrides everything: fleeing, eating, avoiding danger.",
    "CODE": "Ethics. Hard rules it must never break.",
    "AMBITION": "Purpose. What it works toward when it is safe.",
    "TEMPERAMENT": "Personality. How it acts, and what it does when idle.",
}

TEMPLATE = """# The bot's internal manifesto. One statement per category, in plain English.
# Lines starting with # are ignored. Change these and the bot changes who it is.
# Editing this file makes the bot recompile its rules once, which costs about a cent.

# SURVIVAL - instinct. Overrides everything when triggered (fleeing, eating, avoiding danger).
SURVIVAL: {SURVIVAL}

# CODE - ethics. Hard rules it must never break.
CODE: {CODE}

# AMBITION - purpose. The long-term goal it works toward when it is safe.
AMBITION: {AMBITION}

# TEMPERAMENT - personality. How it acts and what it does when idle.
TEMPERAMENT: {TEMPERAMENT}
"""


class Editor:
    def __init__(self, root, manifesto_path, rules_path):
        self.path = manifesto_path
        self.rules_path = rules_path
        self.boxes = {}
        root.title("Manifesto")
        root.columnconfigure(0, weight=1)

        frame = ttk.Frame(root, padding=10)
        frame.grid(sticky="nsew")
        frame.columnconfigure(0, weight=1)
        row = 0
        for name in CATEGORIES:
            ttk.Label(frame, text=name, font=("Segoe UI", 11, "bold")).grid(row=row, column=0, sticky="w")
            ttk.Label(frame, text=BLURB[name], foreground="#666").grid(row=row + 1, column=0, sticky="w")
            box = tk.Text(frame, height=3, wrap="word", font=("Segoe UI", 10))
            box.grid(row=row + 2, column=0, sticky="ew", pady=(2, 10))
            self.boxes[name] = box
            row += 3

        buttons = ttk.Frame(frame)
        buttons.grid(row=row, column=0, sticky="ew")
        ttk.Button(buttons, text="Apply to the bot", command=self.apply).pack(side="left")
        ttk.Button(buttons, text="Reload from file", command=self.load).pack(side="left", padx=6)
        self.status = ttk.Label(frame, text="", foreground="#333", wraplength=520, justify="left")
        self.status.grid(row=row + 1, column=0, sticky="w", pady=(8, 0))

        self.milestones = ttk.Label(frame, text="", foreground="#555", wraplength=520, justify="left")
        self.milestones.grid(row=row + 2, column=0, sticky="w", pady=(8, 0))

        self.load()

    def load(self):
        try:
            statements = parse_manifesto(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            self.say(f"Could not read {self.path.name}: {e}")
            return
        for name, text in statements.items():
            self.boxes[name].delete("1.0", "end")
            self.boxes[name].insert("1.0", text)
        self.say(f"Loaded {self.path.name}.")
        self.show_milestones()

    def show_milestones(self):
        """The ladder the bot worked out from the AMBITION, if it has compiled one."""
        try:
            rules = json.loads(self.rules_path.read_text(encoding="utf-8"))["rules"]
        except (OSError, ValueError, KeyError):
            self.milestones.config(text="")
            return
        steps = rules.get("ambition_milestones") or []
        if steps:
            listed = "\n".join(f"  {i}. {m}" for i, m in enumerate(steps, 1))
            self.milestones.config(text="Milestones it worked out from this ambition:\n" + listed)

    def apply(self):
        statements = {n: " ".join(b.get("1.0", "end").split()) for n, b in self.boxes.items()}
        missing = [n for n, v in statements.items() if not v]
        if missing:
            self.say("Still empty: " + ", ".join(missing))
            return
        try:
            self.path.write_text(TEMPLATE.format(**statements), encoding="utf-8")
        except OSError as e:
            self.say(f"Could not write {self.path.name}: {e}")
            return
        self.say(f"Saved. A running bot picks this up within a couple of seconds, recompiles "
                 f"(about a cent) and starts a new objective. Watch its log for 'Manifesto updated'.")

    def say(self, text):
        self.status.config(text=text)


def run_editor(manifesto_path, rules_path):
    root = tk.Tk()
    Editor(root, manifesto_path, rules_path)
    root.mainloop()
