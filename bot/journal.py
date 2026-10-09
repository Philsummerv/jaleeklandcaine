"""A short-term memory for one run.

Every tactician call is otherwise independent: one screenshot, the objective, and a
fifteen-word note to itself. That is enough to decide the next three seconds and nothing
more, so the bot could spend a minute shuffling around the same canyon with no sense that
it had already been there, or which way it had tried.

The journal is written locally, costs nothing, and is handed back in the state. It is
wiped when a run starts, so it never carries yesterday's wandering into today, and the
file is left on disk afterwards to read.
"""

import threading
import time


class Journal:
    def __init__(self, path=None, keep=14):
        self.path = path
        self.keep = keep
        self.lines = []
        self.started = time.monotonic()
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.lines = []
            self.started = time.monotonic()
        if self.path:
            try:
                open(self.path, "w", encoding="utf-8").close()
            except OSError:
                pass

    def stamp(self):
        seconds = int(time.monotonic() - self.started)
        return f"{seconds // 60:d}:{seconds % 60:02d}"

    def add(self, kind, text):
        """kind is a short tag - plan, goal, hit, stuck - so the model can skim the shape
        of the run rather than reading it as prose."""
        line = f"[{self.stamp()}] {kind}: {' '.join(str(text).split())}"
        with self.lock:
            self.lines.append(line)
            del self.lines[:-200]
        if self.path:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass
        return line

    def recent(self, n=None):
        with self.lock:
            return list(self.lines[-(n or self.keep):])
