"""The hymnal: a fixed catalog of vetted one-bar patterns, one list per layer.

Jev sees only the option names and their descriptions (effects, never "use when").
ENERGY is code-side (0-3) and never sent: it scores how well a run follows the arc.
Patterns are 16 steps per bar. Pitched layers are written relative to the bar's chord.
"""

HARMONY = {  # chord -> (description, bass root midi, voicing midi)
    "Am": ("A minor, the home chord: settled and dark", 33, (57, 60, 64)),
    "F": ("F major: a warm lift away from home", 29, (57, 60, 65)),
    "C": ("C major: bright and open", 36, (55, 60, 64)),
    "G": ("G major: pushes forward and wants to move on", 31, (55, 59, 62)),
    "Dm": ("D minor: soft and melancholy", 38, (57, 62, 65)),
    "Em": ("E minor: tense and unresolved", 28, (55, 59, 64)),
    "E": ("E major: a strong pull straight back to A minor", 28, (56, 59, 64)),
}

# drums: name -> (description, energy, {voice: [steps]})
DRUMS = {
    "rest": ("No drums at all", 0, {}),
    "shaker": ("A soft shaker on the eighth notes, no kick", 1, {"shaker": list(range(0, 16, 2))}),
    "hats": ("Closed hi-hats on every sixteenth, light motion, no kick", 1, {"hat": list(range(16))}),
    "pulse": ("A kick drum on each beat and nothing else", 1, {"kick": [0, 4, 8, 12]}),
    "halftime": ("Kick on beat one, snare on beat three: heavy and slow-feeling", 2,
                 {"kick": [0, 10], "snare": [8], "hat": [0, 4, 8, 12]}),
    "backbeat": ("Kick on one and three, snare on two and four, eighth-note hats", 2,
                 {"kick": [0, 8], "snare": [4, 12], "hat": list(range(0, 16, 2))}),
    "four": ("Four-on-the-floor kick with open hats on the off-beats: dance drive", 3,
             {"kick": [0, 4, 8, 12], "ohat": [2, 6, 10, 14], "snare": [4, 12]}),
    "break": ("Syncopated breakbeat with ghost snares and sixteenth hats", 3,
              {"kick": [0, 3, 10], "snare": [4, 7, 12, 15], "hat": list(range(16))}),
    "stomp": ("A kick on every eighth note with claps: maximum drive", 3,
              {"kick": list(range(0, 16, 2)), "snare": [4, 12], "ohat": [2, 6, 10, 14]}),
    "roll": ("A snare roll that speeds up through the bar and ends on a crash: a fill that announces a change", 2,
             {"snare": [0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15], "kick": [0], "crash": [15]}),
}

# bass: name -> (description, energy, [(step, length_steps, semitones_from_root)])
BASS = {
    "rest": ("No bass", 0, []),
    "drone": ("The chord root held for the whole bar", 1, [(0, 16, 0)]),
    "pulse": ("The root on each beat", 1, [(s, 3, 0) for s in (0, 4, 8, 12)]),
    "sub": ("One deep low note that slides down an octave at the end", 1, [(0, 12, 0), (12, 4, -12)]),
    "walk": ("A walking line: root, third, fifth, then a step toward the next chord", 2,
             [(0, 4, 0), (4, 4, 3), (8, 4, 7), (12, 4, 10)]),
    "syncopated": ("Off-beat stabs that dodge the kick", 2, [(s, 2, 0) for s in (3, 6, 10, 14)] + [(11, 1, 12)]),
    "octaves": ("Root and octave bouncing on eighth notes", 3, [(s, 2, 0 if (s // 2) % 2 == 0 else 12) for s in range(0, 16, 2)]),
    "gallop": ("A driving sixteenth-note gallop on the root", 3,
               [(s, 1, 0) for s in range(16) if s % 4 != 1]),
}

# keys: name -> (description, energy, [(step, length_steps, chord_tone_index or 'chord')])
KEYS = {
    "rest": ("No keys", 0, []),
    "pad": ("A sustained chord that swells in slowly", 1, [(0, 16, "chord")]),
    "stabs": ("Short chord stabs on the off-beats", 2, [(s, 1, "chord") for s in (2, 6, 10, 14)]),
    "pulse": ("The chord pulsing on every eighth note", 2, [(s, 1, "chord") for s in range(0, 16, 2)]),
    "arp_up": ("A rising sixteenth-note arpeggio across two octaves", 3, [(s, 1, s % 6) for s in range(16)]),
    "arp_down": ("A falling sixteenth-note arpeggio across two octaves", 3, [(s, 1, 5 - s % 6) for s in range(16)]),
}

# lead: name -> (description, energy, [(step, length_steps, scale_degrees_from_chord_root)])
LEAD = {
    "rest": ("No melody", 0, []),
    "long": ("One long held note on the fifth", 1, [(0, 16, 4)]),
    "sparkle": ("A few high bell notes, sparse", 1, [(0, 2, 7), (6, 2, 9), (11, 3, 11)]),
    "call": ("A rising four-note phrase that ends like a question", 2, [(0, 3, 0), (4, 3, 2), (8, 3, 4), (12, 4, 5)]),
    "answer": ("A falling phrase that comes to rest on the root", 2, [(0, 3, 4), (4, 3, 3), (8, 2, 1), (10, 6, 0)]),
    "hook": ("The main hook: a catchy syncopated motif", 3,
             [(0, 2, 4), (3, 1, 4), (4, 2, 5), (6, 2, 4), (10, 2, 2), (12, 4, 0)]),
    "run": ("A fast sixteenth-note scale run upward", 3, [(s, 1, s // 2) for s in range(0, 16)]),
}

LAYERS = {"harmony": HARMONY, "drums": DRUMS, "bass": BASS, "keys": KEYS, "lead": LEAD}

# The arc: (first bar, last bar, section, intent shown to Jev, target energy 0-3 per layer on average)
ARC = [
    (1, 4, "intro", "Start from near silence and bring in one element at a time.", 0.5),
    (5, 12, "verse", "Settle into a groove and a repeating chord cycle; leave room, no main hook yet.", 1.5),
    (13, 16, "build", "Rise in intensity every bar; the last bar of the build should announce what is coming.", 2.2),
    (17, 24, "drop", "Everything in at full energy; this is where the main hook belongs.", 2.8),
    (25, 28, "breakdown", "Strip back to a pad and one quiet voice; darker and sparse.", 0.8),
    (29, 31, "finale", "Full energy again for a last run of the hook.", 2.8),
    (32, 32, "outro", "The final bar: land on the home chord and let it ring out.", 0.4),
]
N_BARS = 32


def section(bar):
    for a, b, name, intent, target in ARC:
        if a <= bar <= b:
            return name, intent, target, bar - a + 1, b - a + 1
    raise ValueError(bar)


def energy(choice):
    """Mean code-side energy of the four rhythmic/melodic layers for one bar's choices."""
    return sum(LAYERS[l][choice[l]][1] for l in ("drums", "bass", "keys", "lead")) / 4

# v2: harmony is chosen per 4-bar phrase from vetted progressions (v1 chose a chord per bar and
# collapsed to an Am/E vamp). name -> (description, four chords)
PROGRESSIONS = {
    "home": ("Stay on the home chord for four bars: static and hypnotic", ("Am", "Am", "Am", "Am")),
    "anthem": ("The classic minor lift: hopeful and anthemic", ("Am", "F", "C", "G")),
    "descent": ("A falling line that ends on a strong pull home: dramatic", ("Am", "G", "F", "E")),
    "brood": ("Brooding minor movement: dark and closed", ("Am", "Dm", "Em", "Am")),
    "warm": ("Gentle and warm, settling softly at home", ("F", "C", "Dm", "Am")),
    "tension": ("Rising tension that ends unresolved, wanting to go home", ("Dm", "Em", "F", "E")),
    "cadence": ("A lift and a strong cadence that lands on the home chord", ("F", "G", "E", "Am")),
}
PHRASE = 4
