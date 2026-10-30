"""3-bit remex codes of tomaarsen/embeddinggemma-2-512d embeddings, through sentence-transformers.

Usage: python3 -I run/st_remex_3bit.py      (pip install sentence-transformers remex)

The snippet in RESULTS.md, runnable: an index of 192-byte documents (512 dims x 3 bits) searched
with float queries. RemexIndex calls the model's encode_document / encode_query, so the model's
document and query prompts apply. center=True quantizes relative to the corpus mean, which is what
round 3 measured (NanoBEIR mean NDCG@10 62.86; 62.40 without centering, run/nanobeir_st_check.py).
"""
from sentence_transformers import SentenceTransformer
from remex import RemexIndex

documents = [
    "The northern lights are caused by charged particles from the sun hitting the upper atmosphere.",
    "Photosynthesis turns light, water and carbon dioxide into sugar and oxygen.",
    "A sourdough starter is a culture of wild yeast and lactic acid bacteria.",
    "Auroras are most often seen near the magnetic poles during strong solar wind.",
]

model = SentenceTransformer("tomaarsen/embeddinggemma-2-512d")
index = RemexIndex(model, bits=3, center=True)   # 512 dims x 3 bits = 192 bytes per document
index.add(documents)                              # encode_document, then rotate and quantize
print(index.search("What causes the northern lights?", top_k=2))   # encode_query; the query stays float
