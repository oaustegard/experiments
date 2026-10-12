"""Gemini image generation through the CF gateway, with optional reference images.

  python3 img.py "<prompt>" out.png [ref1.png ...]
"""
import base64, sys, json, time
from pathlib import Path
sys.path.insert(0, "/mnt/skills/user/invoking-gemini/scripts")
import gemini_client as gc

MODEL = gc.IMAGE_MODELS[gc.MODEL_ALIASES["image-pro"]] if "image-pro" in gc.MODEL_ALIASES else "gemini-3-pro-image-preview"

STYLE = ("Oil painting in the manner of a Swiss realist painter of the 1890s, in the spirit of Félix Vallotton's "
         "interiors and Albert Anker's portraits: muted earth and slate palette, lamplight and deep shadow, "
         "firm drawing, flat calm surfaces, visible but restrained brushwork. Cinematic 16:9 composition. "
         "No text, no lettering, no signatures, no watermarks, no modern objects.")

def generate(prompt, out, refs=(), temperature=0.8, style=True):
    parts = []
    for r in refs:
        parts.append({"inlineData": {"mimeType": "image/png", "data": base64.b64encode(Path(r).read_bytes()).decode()}})
    text = (prompt + "\n\n" + STYLE) if style else prompt
    if refs:
        text = ("The attached images are character reference portraits; keep each character's face, age, hair and clothing "
                "consistent with them.\n\n" + text)
    parts.append({"text": text})
    cfg = {"temperature": temperature, "responseModalities": ["IMAGE", "TEXT"], "imageConfig": {"aspectRatio": "16:9"}}
    creds = gc.get_cf_credentials()
    for attempt in range(4):
        try:
            r = gc._cf_request(MODEL, [{"parts": parts}], cfg, creds)
            for p in r["candidates"][0]["content"]["parts"]:
                d = p.get("inlineData") or p.get("inline_data")
                if d:
                    Path(out).parent.mkdir(parents=True, exist_ok=True)
                    Path(out).write_bytes(base64.b64decode(d["data"])); return out
            raise RuntimeError("no image in response: " + json.dumps(r)[:300])
        except Exception as e:
            err = e; time.sleep(3 * (attempt + 1))
    raise SystemExit(f"failed {out}: {err}")

if __name__ == "__main__":
    print(generate(sys.argv[1], sys.argv[2], sys.argv[3:]))
