"""Timeline-to-CSS compiler for animated SVG, after svjeez's svg-motion.

A Motion is a sequence of states on an absolute clock: every state sets every
channel, so one Motion compiles to one @keyframes block. A Sheet interns
keyframe bodies and animation rules by their exact text, so elements that
move alike share one rule however many of them there are.

    m = Motion(opacity=0).at(2).to(0.6, "out", opacity=1)
    sheet = Sheet(duration=48)
    cls = sheet.cls(m)          # "m0"; the CSS lives in sheet.css()
"""

EASE = {
    "linear": "linear",
    "in": "cubic-bezier(.55,0,1,.45)",
    "out": "cubic-bezier(.23,1,.32,1)",
    "inOut": "cubic-bezier(.77,0,.175,1)",
    "pop": "cubic-bezier(.34,1.56,.64,1)",
    "sine": "cubic-bezier(.37,0,.63,1)",
}


def num(v, places=3):
    """Shortest text for a number: 3 places, no trailing zeros, no leading 0."""
    if isinstance(v, str):
        return v
    t = f"{v:.{places}f}".rstrip("0").rstrip(".")
    if t.startswith("0.") and len(t) > 2:
        t = t[1:]
    elif t.startswith("-0.") and len(t) > 3:
        t = "-" + t[2:]
    return "0" if t in ("", "-0", "-") else t


class Motion:
    """States at absolute times. Each segment's easing sits on its first state."""

    def __init__(self, **start):
        self.points = [[0.0, dict(start), None]]

    @property
    def now(self):
        return self.points[-1][0]

    def at(self, t):
        """Hold the current values until time t."""
        if t < self.now - 1e-9:
            raise ValueError(f"at({t}) goes back before {self.now}")
        if t > self.now + 1e-9:
            self.points.append([t, dict(self.points[-1][1]), None])
        return self

    def to(self, dur, ease="linear", **values):
        """Move to values over dur seconds, starting now."""
        self.points[-1][2] = ease
        self.points.append([self.now + dur, {**self.points[-1][1], **values}, None])
        return self

    def keyframes(self, duration):
        """Keyframe text body, e.g. '0%,40%{opacity:0}50%{opacity:1}'."""
        if self.now > duration + 1e-9:
            raise ValueError(f"motion runs to {self.now}s, past the {duration}s loop")
        pts = [list(p) for p in self.points]
        if pts[-1][0] < duration - 1e-9:
            pts.append([duration, dict(pts[-1][1]), None])
        groups = {}  # declaration text -> percentages, in first-seen order
        for t, vals, ease in pts:
            decl = ";".join(f"{k}:{num(v)}" for k, v in vals.items())
            if ease and ease != "linear":
                decl += f";animation-timing-function:{EASE.get(ease, ease)}"
            groups.setdefault(decl, []).append(num(100 * t / duration) + "%")
        return "".join(f"{','.join(p)}{{{d}}}" for d, p in groups.items())


class Sheet:
    """Interns @keyframes and animation classes; emits one compact CSS string."""

    def __init__(self, duration, prefix="m"):
        self.duration = duration
        self.prefix = prefix
        self.frames = {}   # keyframes body -> name
        self.rules = {}    # rule body -> class name

    def cls(self, motion, duration=None, delay=0):
        """Class name whose rule plays motion. Equal motions share a class."""
        d = duration or self.duration
        body = motion.keyframes(d)
        k = self.frames.setdefault(body, f"k{self.prefix}{len(self.frames)}")
        rule = f"animation-name:{k}"
        if d != self.duration:
            rule += f";animation-duration:{num(d)}s"
        if delay:
            rule += f";animation-delay:{num(delay)}s"
        return self.rules.setdefault(rule, f"{self.prefix}{len(self.rules)}")

    def css(self):
        out = [f"@keyframes {k}{{{b}}}" for b, k in self.frames.items()]
        out += [f".{c}{{{r}}}" for r, c in self.rules.items()]
        return "\n".join(out)
