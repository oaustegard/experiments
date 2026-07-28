"""
SPECTER2 embedding gap fill for remex#69 phase-0.

Embeds three papers that are not yet in Semantic Scholar's precomputed
SPECTER2 index (too recent / not indexed at the relevant version) using
allenai/specter2_base. Writes 768-d vectors as JSON.

Input papers (see issue oaustegard/claude-workspace#87):
  - sawin              arXiv:2605.20579  Will Sawin, unit-distance explicit bound
  - openai_companion   arXiv:2605.20695  Alon-Bloom-Gowers et al., remarks
  - lenstra_1986       Lenstra 1986, "Codes from algebraic number fields"
                       (CWI Monographs IV) — abstract elided by publisher
                       in Semantic Scholar; PDF host unreachable from CCotw.
                       Fallback per the issue: use a modern AG-codes survey
                       intro that explicitly cites Lenstra 1986 as the
                       originating class-field-theory-to-codes bridge.
                       We use the Maire-Oggier 2017 abstract ("Maximal order
                       codes over number fields", J. Pure & Appl. Algebra),
                       which opens with "Particular cases include codes from
                       algebraic number fields by Lenstra and Guruswami...".
"""

import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

OUT_PATH = Path("/home/claude/sawin_lenstra_specter.json")
OUT_PATH_LOCAL = Path(__file__).resolve().parent / "sawin_lenstra_specter.json"

PAPERS = {
    "sawin": {
        "source_id": "arXiv:2605.20579",
        "title": "An explicit lower bound for the unit distance problem",
        "abstract": (
            "We show that there are sets of n points in the plane with n "
            "arbitrarily large that contain more than n^{1.014} pairs of "
            "points separated by a distance exactly 1. This improves on "
            "very recent work of a team at OpenAI, who proved the same "
            "result with an inexplicit exponent greater than 1, drastically "
            "improving on the best previous lower bound and disproving a "
            "conjecture of Erdős. The method is number-theoretic, relying "
            "on constructing algebraic number fields of large degree and "
            "small discriminant with many primes of small norm via a "
            "Golod-Shafarevich criterion argument."
        ),
        "abstract_source": "issue body (verbatim from arXiv:2605.20579)",
    },
    "openai_companion": {
        "source_id": "arXiv:2605.20695",
        "title": "Remarks on the disproof of the unit distance conjecture",
        "abstract": (
            "We present a short, digested, human-verified version of the "
            "recent OpenAI-generated counterexample to the Erdős unit "
            "distance conjecture, and a sequence of reflections on it. "
            "The argument relies crucially on ideas that may, at least in "
            "retrospect, be attributed to Ellenberg-Venkatesh, "
            "Golod-Shafarevich, and Hajir-Maire-Ramakrishna."
        ),
        "abstract_source": "arXiv API id_list=2605.20695 (fetched 2026-05-23)",
    },
    "lenstra_1986": {
        "source_id": "SemanticScholar:e1694ecd50ec042444acaa59921cc2aedaa01ef9",
        "title": "Codes from algebraic number fields",
        "abstract": (
            "Abstract We present constructions of codes obtained from "
            "maximal orders over number fields. Particular cases include "
            "codes from algebraic number fields by Lenstra and Guruswami, "
            "codes from units of the ring of integers of number fields, "
            "and codes from both additive and multiplicative structures "
            "of maximal orders in central simple division algebras. The "
            "parameters of interest are the code rate and the minimum "
            "Hamming distance. An asymptotic study reveals several "
            "families of asymptotically good codes."
        ),
        "abstract_source": (
            "Fallback per issue: Lenstra 1986's own abstract is elided by "
            "the publisher in Semantic Scholar and the Leiden Open Access "
            "PDF host is unreachable from CCotw. Substituted the "
            "Maire-Oggier 2017 abstract ('Maximal order codes over number "
            "fields', J. Pure & Appl. Algebra, "
            "doi:10.1016/J.JPAA.2017.08.009, SS paperId "
            "92400b837574c97a4bf990abfc01f871301879b7), which is a modern "
            "AG-codes survey whose first sentence explicitly cites "
            "'codes from algebraic number fields by Lenstra' as the "
            "originating construction. Title is Lenstra's original."
        ),
    },
}


def main() -> None:
    print("Loading allenai/specter2_base...")
    tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoModel.from_pretrained("allenai/specter2_base")
    model.eval()

    out: dict = {}
    for label, paper in PAPERS.items():
        text = paper["title"] + tok.sep_token + paper["abstract"]
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        n_tokens = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
        vector = hidden[:, 0, :].squeeze().tolist()
        assert len(vector) == 768, f"{label}: expected 768-d, got {len(vector)}"
        print(f"  {label}: {n_tokens} tokens -> 768-d vector "
              f"(L2={sum(v * v for v in vector) ** 0.5:.3f})")
        out[label] = {
            "vector": vector,
            "title": paper["title"],
            "source_id": paper["source_id"],
            "abstract_source": paper["abstract_source"],
            "n_input_tokens": n_tokens,
        }

    payload = {
        "model": "allenai/specter2_base",
        "dim": 768,
        "pooling": "cls",
        "papers": out,
    }
    blob = json.dumps(payload, indent=2)
    OUT_PATH_LOCAL.write_text(blob)
    print(f"Wrote {OUT_PATH_LOCAL} ({len(blob)} bytes)")
    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(blob)
        print(f"Wrote {OUT_PATH} ({len(blob)} bytes)")
    except OSError as e:
        print(f"Skipped {OUT_PATH}: {e}")


if __name__ == "__main__":
    main()
