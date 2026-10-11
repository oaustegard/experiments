"""Phase 0: brainstorm constellation concepts per house, so 140 parallel agents don't all draw 'the Great Kite'."""
import json, pathlib
BIBLE = open("bible.md").read()
HOUSES = ["The Spindle", "The Windward Reach", "The Lamp Road", "The Leeward Reach", "The Unflown"]
ANGLES = {"objects": "tools, vessels, garments, household and craft objects, kites of particular kinds",
          "creatures": "animals, birds, sea creatures, monsters, plants and weather-beings",
          "people": "named mortals, ancestors, scenes and episodes (someone doing something specific), and parts of the gods' stories"}
schema = {"type": "object", "properties": {"concepts": {"type": "array", "items": {"type": "object", "properties": {
    "gloss": {"type": "string", "description": "English name, 2-5 words, e.g. 'The Split Oar'"},
    "hook": {"type": "string", "description": "one sentence: who/what it is and why it is in the sky"}},
    "required": ["gloss", "hook"]}}}, "required": ["concepts"]}
with open("work/t0.jsonl", "w") as f:
    for h in HOUSES:
        for a, desc in ANGLES.items():
            f.write(json.dumps(dict(id=f"{h.split()[1].lower()}-{a}", system="You invent the constellations of a fictional culture. Follow the world bible exactly.",
                schema=schema, prompt=f"{BIBLE}\n\n---\nBrainstorm 30 distinct constellation concepts for **{h}**, drawn from: {desc}. "
                "Each must fit that house's character and the bible's register. Make them specific and varied (not 'a kite' but 'the kite flown for a drowned child'). "
                "Avoid any echo of Western/Greek constellations.")) + "\n")
