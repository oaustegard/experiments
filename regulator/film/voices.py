"""Design the cast's voices (POST v1beta/voices, type "prompted") and store ids in voices.json."""
import json, os, sys, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "muninn-story"))
sys.path.insert(0, "/mnt/skills/user/invoking-gemini/scripts")
import gemini_client as gc
GENDER = dict(M="male", O="female", F="male", L="female", D="male", B="male", R="female")

CAST = {
 "M": "Inspector Léon Matthey, a Swiss detective of about sixty-five remembering a case from his forties: a warm, low, unhurried male voice, educated, speaking English with a light French-Swiss accent. Dry humour, patient, precise, never theatrical. Close to the microphone, like a man telling a story across a table.",
 "O": "Odile Monnier, a composed widow of fifty-five from Neuchâtel: a cool, clear, low female voice, very precise diction, speaking English with a light French-Swiss accent. Short sentences, controlled, cold irony, no tears.",
 "F": "Félix Monnier, a charming, frightened man of twenty-eight: a light, quick male tenor, speaking English with a faint French accent, talks a little too fast, nervous humour, slightly drunk on brandy.",
 "L": "Lina Ruedi, a régleuse of twenty-four, a farmer's daughter from the Jura: a young female voice, flat and exact, steady and quiet, speaking English with a light French-Swiss accent. Speaks in measurements and keeps feeling out of her voice.",
 "D": "Dr Casimir Vauthier, an urbane physician of fifty-eight: a rich, warm, cultivated baritone, speaking English with a light French accent, genial and reassuring, a man who enjoys his own sentences, a cigar-smoker's softness.",
 "B": "Bastien Girard, a big blunt merchant of forty-five from Le Locle: a gruff, heavy male voice, speaking English with a rough French-Swiss accent, impatient, commercial, plain-spoken.",
 "R": "Rosa Häfeli, a maid and cook of forty from the Emmental: a warm, plain female voice, speaking English with a Swiss-German accent, frightened but proud, practical, a little breathless.",
}

def design(key, prompt):
    return gc.design_voice(prompt, f"regulator-{key}", gender=GENDER[key], language_code="en-GB")

if __name__ == "__main__":
    out = json.load(open("voices.json")) if Path("voices.json").exists() else {}
    keys = sys.argv[1:] or list(CAST)
    for k in keys:
        if k in out: continue
        r = design(k, CAST[k]); print(k, r)
        if r and r["id"]:
            out[k] = r["id"]; json.dump(out, open("voices.json", "w"), indent=1)
            if r["sample_path"]: os.replace(r["sample_path"], f"samples/{k}.wav")
