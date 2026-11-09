"""Prompts for Arm B/C. The problem text is Arm A's system message verbatim.

Focus axes are the 14 aspects and 14 perspectives from SwarmEvolve
(arXiv 2611.08483, Supplementary Table 4). The paper names two of its nine
thinking styles and one phrasing per axis; the other seven styles here are
ours.
"""

PROBLEM = """You are an expert mathematician specializing in circle packing problems and computational geometry. Your task is to improve a constructor function that directly produces a specific arrangement of 26 circles in a unit square, maximizing the sum of their radii. The AlphaEvolve paper achieved a sum of 2.635 for n=26.

Key geometric insights:
- Circle packings often follow hexagonal patterns in the densest regions
- Maximum density for infinite circle packing is pi/(2*sqrt(3)) ≈ 0.9069
- Edge effects make square container packing harder than infinite packing
- Circles can be placed in layers or shells when confined to a square
- Similar radius circles often form regular patterns, while varied radii allow better space utilization
- Perfect symmetry may not yield the optimal packing due to edge effects

Focus on designing an explicit constructor that places each circle in a specific position, rather than an iterative search algorithm.
"""

INTERFACE = """The program is a complete Python file. The evaluator imports it and calls `run_packing()`, which must return `(centers, radii, sum_radii)`: centers a numpy array of shape (26, 2), radii of shape (26,), all circles inside the unit square and none overlapping. An invalid packing scores 0. numpy and scipy are available."""

ASPECTS = [
    "The starting point of the solution",
    "The optimization or solution algorithm",
    "Global exploration, restarts, and annealing",
    "Symmetries and structure of good solutions",
    "Feasibility, tolerances, and a final repair step",
    "The use of the time limit",
    "The largest source of the remaining error",
    "A smoother surrogate or a penalty formulation",
    "The weakest part of the current solution",
    "What the best programs in the field do differently",
    "Bugs, wasted computation, and silent failures",
    "Simpler and more robust methods",
    "The numbers in the program",
    "The formulation and its free variables",
]

PERSPECTIVES = [
    "The standard textbook method",
    "A fundamentally different algorithm",
    "A known construction from the literature",
    "A physics-inspired method",
    "A parametrization that satisfies the constraints",
    "Many cheap starts of one fast local optimizer",
    "A symmetry that forces structure",
    "A coarse-to-fine or incremental construction",
    "A greedy construction with local improvement",
    "Linear or convex programming for convex parts",
    "An exact or analytical treatment",
    "Sparsity, separability, or low rank",
    "A higher-order or adaptive method",
    "An idea that nobody would try first",
]

STYLES = [
    "Reason from first principles.",                       # paper
    "Question one assumption the program makes.",          # paper
    "Work backwards from what an optimal answer must look like.",
    "Look for the single change with the largest expected gain.",
    "Reason by analogy to a related problem you know well.",
    "Look for where the program fails or wastes its effort.",
    "Estimate the size of the gain before you propose it.",
    "Remove or simplify something rather than add something.",
    "Combine two ideas that are already in the program.",
]

IDEA_WORDS = 45  # ~60 tokens; ideas are cut to this many words before the synthesizer sees them


def _texts(spec):
    if spec is None:
        return PROBLEM, INTERFACE, "score (sum_radii / 2.635983)", ("sum_radii", "validity", "eval_time"), "run_packing"
    return spec["system_message"], spec["interface"], spec["score_label"], spec["show"], spec["entry"]


def program_block(p: dict, label: str, spec: dict | None = None) -> str:
    m = p.get("metrics") or {}
    _, _, score_label, show, _ = _texts(spec)
    lines = [f"### {label}", f"{score_label}: {p.get('score', 0.0):.6f}"]
    for k in show:
        if k in m:
            lines.append(f"{k}: {m[k]}")
    if m.get("error"):
        lines.append(f"error: {str(m['error'])[:800]}")
    lines += ["```python", p["code"].rstrip(), "```"]
    return "\n".join(lines)


def proposer_system(parent: dict | None, spec: dict | None = None) -> str:
    problem, interface, *_ = _texts(spec)
    head = ("You are one member of a research team improving a program. Teammates each "
            "propose one idea from a different angle; a lead programmer then writes the "
            "next version from the ideas.\n\n## Problem\n" + problem + "\n" + interface)
    if parent is None:
        return head + "\n\nThere is no program for you to read: propose from the problem alone."
    return head + "\n\n## Program\n" + program_block(parent, "Current program", spec)


def proposer_user(focus_phrase: str, focus: str, style: str) -> str:
    return (f"{focus_phrase} {focus.lower()}. {style}\n\n"
            "Propose ONE concrete change that would raise the score. Reply with the idea "
            f"only, in at most {IDEA_WORDS} words: no code, no preamble.")


def synth_system(parents: list[dict], spec: dict | None = None) -> str:
    problem, interface, *_ = _texts(spec)
    s = ("You are the lead programmer of a research team improving a program.\n\n## Problem\n"
         + problem + "\n" + interface)
    if parents:
        s += "\n\n## Programs\n" + "\n\n".join(program_block(p, f"Program {chr(65 + i)}", spec)
                                           for i, p in enumerate(parents))
    return s


def synth_user(ideas: list[str], n_parents: int, spec: dict | None = None) -> str:
    entry = _texts(spec)[4]
    if ideas:
        body = "Your teammates proposed these ideas:\n\n" + "\n".join(
            f"{i + 1}. {t}" for i, t in enumerate(ideas))
        body += "\n\nChoose the most promising ideas, combine them where they fit, and write the next version."
    else:
        body = "Write the next version of the program, improving its score."
    if n_parents > 1:
        body += " Draw on the best parts of each program above."
    return (body + "\n\nReply with at most two sentences describing your direction, then the "
            "complete program in a single ```python block. The program must define "
            f"{entry}() as specified.")
