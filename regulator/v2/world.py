"""The Regulator — the formal case.

Everything a reader can rely on is a statement here: an author, a plain-English
gloss, and the constraints it asserts. The prose interviews are written from
these and checked back against them; the solver in verify.py proves the answer
is unique from them alone.

Time is minutes after 20:00 on 31 December 1899. A clock or watch reading r of
an event e at offset o satisfies r = t(e) + o (o > 0 runs fast). The temple
bell is true time. The atelier regulator was stopped and its hands moved by the
murderer, so its final face is physical evidence, not a statement.
"""

def m(hhmm):
    h, mm = map(int, hhmm.split(":"))
    if h < 12:
        h += 24
    return (h - 20) * 60 + mm


def hhmm(x):
    h, mm = divmod(x, 60)
    return f"{(h + 20) % 24:02d}:{mm:02d}"


PEOPLE = {
    "O": "Odile Monnier",
    "F": "Félix Monnier",
    "L": "Lina Ruedi",
    "D": "Dr Casimir Vauthier",
    "B": "Bastien Girard",
    "R": "Rosa Häfeli",
}
VICTIM = "Aurèle Monnier"

ROOMS = {
    "atelier": "the atelier (top floor, under the long windows)",
    "library": "the library (first floor, directly below the atelier)",
    "salon": "the salon (ground floor, front)",
    "dining": "the dining room (ground floor)",
    "hall": "the hall (ground floor, stairs, the Morbier clock)",
    "kitchen": "the kitchen (ground floor, back)",
    "street": "the street (Rue du Progrès, under the front windows)",
}

CLOCKS = {
    "salon": "the salon clock (marble mantel clock)",
    "kitchen": "the kitchen clock (American alarm clock)",
    "morbier": "the hall clock (Morbier long-case; strikes the hours by its own face)",
    "regulator": "the atelier regulator",
    "wF": "Félix's pocket watch",
    "wB": "Bastien's pocket watch",
    "wD": "the doctor's pocket watch",
    "wL": "Lina's pocket watch",
}

# Ground truth. Offsets: shows minus true, in minutes.
TRUE_OFF = dict(salon=-5, kitchen=15, morbier=12, regulator=0, wF=-20, wB=-4, wD=2, wL=0)

EVENTS = {
    "bell9":     ("the temple bell strikes nine, during dinner", m("21:00")),
    "dinner_end": ("dinner ends", m("21:12")),
    "bell10":    ("the temple bell strikes ten", m("22:00")),
    "l_down":    ("Lina leaves the atelier", m("22:20")),
    "l_kitchen": ("Lina reaches the kitchen", m("22:22")),
    "b_out":     ("Bastien goes out into the street", m("22:25")),
    "d_out":     ("the doctor leaves the salon", m("22:33")),
    "p_start":   ("the piano begins", m("22:33")),
    "steps_up":  ("footsteps go up past the library door", m("22:35")),
    "chime":     ("the repeater chimes in the atelier (the attack)", m("22:38")),
    "steps_down": ("footsteps come down past the library door", m("22:41")),
    "d_back":    ("the doctor returns to the salon", m("22:46")),
    "p_stop":    ("the piano stops", m("22:46")),
    "morb11":    ("the Morbier strikes eleven", m("22:48")),
    "b_in":      ("Bastien comes back into the house", m("22:50")),
    "b_lib":     ("Bastien enters the library", m("22:52")),
    "bell11":    ("the temple bell strikes eleven", m("23:00")),
}

# Atoms:
#   ("read", event, clock, "HH:MM")      reading of clock at event
#   ("true", event, "HH:MM")             event at a true time (temple bell)
#   ("gap", e1, e2, lo, hi)              lo <= t(e2) - t(e1) <= hi (hi None = open)
#   ("offdiff", a, b, d)                 off(a) - off(b) == d
#   ("at", person, room, e1, e2)         person in room throughout [t(e1), t(e2)]
#   ("during", e, e1, e2)                t(e1) < t(e) < t(e2)
S = []


def st(sid, who, gloss, *atoms, lie=False):
    S.append(dict(id=sid, who=who, gloss=gloss, atoms=list(atoms), lie=lie))


# Premise facts (author None) are the inspector's findings and the bell.
st("P1", None, "The temple bell of La Chaux-de-Fonds strikes the true hour; it struck nine, ten and eleven.",
   ("true", "bell9", "21:00"), ("true", "bell10", "22:00"), ("true", "bell11", "23:00"))
st("P2", None, "Dinner ended and everyone left the dining room before the bell struck ten.",
   ("gap", "bell9", "dinner_end", 1, None), ("gap", "dinner_end", "bell10", 1, None))

# ---- Odile Monnier, the widow
st("O1", "O", "At dinner, when the bell struck nine, Félix's watch showed twenty to nine; Aurèle scolded him for it.",
   ("read", "bell9", "wF", "20:40"))
st("O2", "O", "The temple struck ten while she and the doctor were at piquet in the salon.",
   ("during", "bell10", "dinner_end", "d_out"))
st("O3", "O", "She and the doctor played piquet in the salon from the end of dinner until he went out for his cigars; he went out at 10:28 by the salon clock.",
   ("at", "O", "salon", "dinner_end", "d_out"), ("at", "D", "salon", "dinner_end", "d_out"),
   ("read", "d_out", "salon", "22:28"))
st("O4", "O", "As he went he asked her to play for him; she began at once and played until he came back, and stopped then.",
   ("gap", "d_out", "p_start", 0, 0), ("gap", "d_back", "p_stop", 0, 0), ("at", "O", "salon", "d_out", "d_back"))
st("O5", "O", "A moment after he came back, the hall clock struck eleven.",
   ("gap", "d_back", "morb11", 0, 3), ("read", "morb11", "morbier", "23:00"))

# ---- Félix Monnier, the nephew
st("F1", "F", "His watch was twenty minutes behind at dinner; his uncle made a scene over it.",
   ("read", "bell9", "wF", "20:40"))
st("F2", "F", "He went up to the library after dinner and stayed there alone.",
   ("at", "F", "library", "dinner_end", "b_lib"))
st("F3", "F", "At 10:18 by his watch he heard a thump overhead and then his uncle's repeater: ten slow deep strokes, then doubled notes, then quick high ones he did not count.",
   ("read", "chime", "wF", "22:18"))
st("F4", "F", "Before the thump, at 10:15 by his watch, someone went quickly up the stairs past the library door; at 10:21 someone came down.",
   ("read", "steps_up", "wF", "22:15"), ("read", "steps_down", "wF", "22:21"))
st("F5", "F", "His aunt's piano was playing downstairs when the thump came.",
   ("during", "chime", "p_start", "p_stop"))
st("F6", "F", "Bastien came into the library at 10:32 by his watch and they sat together until the temple struck eleven.",
   ("read", "b_lib", "wF", "22:32"), ("at", "F", "library", "b_lib", "bell11"), ("at", "B", "library", "b_lib", "bell11"))

# ---- Lina Ruedi, the régleuse
st("L1", "L", "At dinner, when the bell struck nine, the master made everyone show their watches: hers was exact, the doctor's two minutes ahead; Bastien would not show his.",
   ("read", "bell9", "wL", "21:00"), ("read", "bell9", "wD", "21:02"))
st("L2", "L", "She worked with the master in the atelier after dinner. When the temple struck ten the regulator agreed with it exactly.",
   ("at", "L", "atelier", "dinner_end", "l_down"), ("read", "bell10", "regulator", "22:00"))
st("L3", "L", "He sent her down when the regulator showed 10:20; it was running, and he was well.",
   ("read", "l_down", "regulator", "22:20"), ("gap", "l_down", "chime", 1, None))
st("L4", "L", "In the kitchen she compared her watch with Rosa's clock: the clock was exactly fifteen minutes fast.",
   ("offdiff", "kitchen", "wL", 15))
st("L5", "L", "From the kitchen they heard Madame's piano begin at 10:48 by the kitchen clock and stop at 11:01.",
   ("read", "p_start", "kitchen", "22:48"), ("read", "p_stop", "kitchen", "23:01"))
st("L6", "L", "She stayed in the kitchen with Rosa from when she came down until after the temple struck eleven.",
   ("at", "L", "kitchen", "l_kitchen", "bell11"), ("at", "R", "kitchen", "l_kitchen", "bell11"),
   ("gap", "l_down", "l_kitchen", 1, 4))

# ---- Rosa Häfeli, the maid
st("R1", "R", "When the temple struck ten, her kitchen clock showed a quarter past; she keeps it fast on purpose.",
   ("read", "bell10", "kitchen", "22:15"))
st("R2", "R", "Mademoiselle Lina came down at 10:37 by the kitchen clock and stayed with her until after the temple struck eleven.",
   ("read", "l_kitchen", "kitchen", "22:37"),
   ("at", "L", "kitchen", "l_kitchen", "bell11"), ("at", "R", "kitchen", "l_kitchen", "bell11"))
st("R3", "R", "The piano began at 10:48 by her clock and stopped at 11:01.",
   ("read", "p_start", "kitchen", "22:48"), ("read", "p_stop", "kitchen", "23:01"))
st("R4", "R", "She heard the hall clock strike eleven when her clock showed 11:03.",
   ("read", "morb11", "kitchen", "23:03"), ("read", "morb11", "morbier", "23:00"))

# ---- Bastien Girard, the établisseur
st("B1", "B", "He would not show his watch at dinner. He sat on in the dining room after dinner and heard the temple strike ten there.",
   ("at", "B", "dining", "dinner_end", "b_out"))
st("B2", "B", "He went out into the street to smoke at 10:21 by his watch, and stayed under the front windows.",
   ("read", "b_out", "wB", "22:21"), ("at", "B", "street", "b_out", "b_in"))
st("B3", "B", "The piano started in the salon as he lit his cigar, at 10:29 by his watch, and stopped at 10:42.",
   ("read", "p_start", "wB", "22:29"), ("read", "p_stop", "wB", "22:42"))
st("B4", "B", "At 10:34 by his watch he heard faint quick bells from the master's window above, while the piano played.",
   ("read", "chime", "wB", "22:34"), ("during", "chime", "p_start", "p_stop"))
st("B5", "B", "He came in at 10:46 by his watch; the hall clock had just struck eleven and showed two minutes past. He went up to Félix in the library.",
   ("read", "b_in", "wB", "22:46"), ("read", "b_in", "morbier", "23:02"), ("read", "morb11", "morbier", "23:00"),
   ("gap", "morb11", "b_in", 0, 3), ("gap", "b_in", "b_lib", 1, 4))

# ---- Dr Casimir Vauthier, the physician (the murderer)
st("D1", "D", "At dinner his watch was two minutes ahead of the bell.",
   ("read", "bell9", "wD", "21:02"))
st("D2", "D", "He played piquet with Madame until 10:45 by his watch, when he went down to the hall for his cigar case. Madame began to play as he went out and stopped when he came back.",
   ("at", "D", "salon", "dinner_end", "d_out"), ("read", "d_out", "wD", "22:45"),
   ("gap", "d_out", "p_start", 0, 0), ("gap", "d_back", "p_stop", 0, 0), lie=True)
st("D3", "D", "When the temple struck ten during the game, the salon clock showed five to ten; he remarked on it.",
   ("read", "bell10", "salon", "21:55"))

SOLUTION = dict(murderer="D", attack=m("22:38"))

PHYSICAL = [
    "The regulator in the atelier has stopped. Its hands stand at five past ten; the pendulum hangs still.",
    "Monnier's new repeater is in his right hand, its slide pressed home. A minute repeater sounds the time on demand: one low note for each hour, a doubled note for each quarter past, a high note for each minute past the quarter.",
    "He was struck from behind with the brass mallet from his own bench, and he pressed the repeater as he fell. Whoever struck him was in the atelier when it sounded.",
]
