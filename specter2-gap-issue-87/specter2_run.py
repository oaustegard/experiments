"""
SPECTER2 embeddings for remex#69 phase-0 gap fill.
Papers: Sawin 2605.20579, OpenAI companion 2605.20695, Lenstra 1986.
"""
import json, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

HERE = Path(__file__).resolve().parent

papers = {
    "sawin": {
        "title": "An explicit lower bound for the unit distance problem",
        "source_id": "arXiv:2605.20579",
        "abstract": (
            "We show that there are sets of n points in the plane with n arbitrarily large "
            "that contain more than n^{1.014} pairs of points separated by a distance exactly 1. "
            "This improves on very recent work of a team at OpenAI, who proved the same result "
            "with an inexplicit exponent greater than 1, drastically improving on the best previous "
            "lower bound and disproving a conjecture of Erdős. The method is number-theoretic, "
            "relying on constructing algebraic number fields of large degree and small discriminant "
            "with many primes of small norm via a Golod-Shafarevich criterion argument."
        ),
    },
    "openai_companion": {
        "title": "Remarks on the disproof of the unit distance conjecture",
        "source_id": "arXiv:2605.20695",
        "abstract": (
            "We present a short, digested, human-verified version of the recent OpenAI-generated "
            "counterexample to the Erdős unit distance conjecture, and a sequence of reflections "
            "on it. The argument relies crucially on ideas that may, at least in retrospect, be "
            "attributed to Ellenberg-Venkatesh, Golod-Shafarevich, and Hajir-Maire-Ramakrishna."
        ),
    },
    "lenstra_1986": {
        "title": "Codes from algebraic number fields",
        "source_id": "Lenstra 1986 (CWI Monographs vol 4)",
        "abstract": (
            "We show how algebraic number fields can be used to construct long linear codes over "
            "finite fields with good parameters. The construction uses the ring of integers of a "
            "number field and its reduction modulo prime ideals. Sequences of number fields with "
            "many small primes and large degree, obtained via class field towers from "
            "Golod-Shafarevich, yield families of codes asymptotically meeting the "
            "Gilbert-Varshamov bound. This establishes a bridge between class field theory and "
            "coding theory, showing that the algebraic geometry of number fields—in particular "
            "the distribution of prime ideals of small norm—directly governs the information-"
            "theoretic capacity of the resulting codes."
        ),
    },
}

print("Loading SPECTER2 tokenizer and model (allenai/specter2_base)...")
tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoModel.from_pretrained("allenai/specter2_base")
model.eval()
print("Model loaded.")

def specter2_embed(title, abstract):
    text = title + tok.sep_token + abstract
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze().tolist()

results = {}
for label, p in papers.items():
    print(f"Embedding: {label}...")
    vec = specter2_embed(p["title"], p["abstract"])
    assert len(vec) == 768, f"Expected 768 dims, got {len(vec)}"
    results[label] = {
        "vector": vec,
        "title": p["title"],
        "source_id": p["source_id"],
    }
    print(f"  -> {len(vec)}-d vector, norm={sum(x**2 for x in vec)**0.5:.4f}")

out_path = HERE / "sawin_lenstra_specter.legacy.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {out_path}")
print("Labels:", list(results.keys()))
