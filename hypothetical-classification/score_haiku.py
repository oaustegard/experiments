"""Score the Haiku-subagent hallucinator against Gemini flash-lite on the same 40."""
import json, sys, numpy as np
sys.path.insert(0,'/home/user/muninn-utilities')
import bench
from muninn_utils.hypothetical_classifier import Vocabulary

HAIKU = """Hydraulic Styling Thrones|Connected Tabletop Hubs|Extinct Creature Shelf Companions|Tropical Tide Accent Throws|Oversized Single Recline Lounges|Paired Comfort Bundles|Transparent Modern Seating|Weathered Branch-Frame Reflectors|Sentimental Entry Plaques|Tabletop Flame Focal Points|Four-Column Statement Beds|Gradient-Blend Floor Coverings|Giant Utensil Wall Sculptures|Patio Screening Barriers|Glow-Frame Sleep Furniture|Five-Section Storage Towers|Square Modular Floor Poufs|Zippered Garment Enclosures|Multi-Bulb Mirror Sconces|Wheeled Patient Transfer Beds|Masonry-Textured Water Features|Pearlescent Light-Control Drapes|Dual-Tone Contrast Cushions|Mixed-Material Side Tables|Tiered Timber Table Suites|Botanical Canvas Wall Art|Premium Hide Seating Furniture|Farmstead Bird Sculptures|Sink Control Handle Hardware|Compact Grip Hardware Handles|Rust-Tone Privacy Drapes|Charcoal Drawer Storage Units|Textured Anti-Skid Bath Surfaces|Supported Counter-Height Seating|Sheltered Footwear Organizers|Rolling Hamper Collection Bins|Mid-Height Counter Bar Chairs|Weather-Resistant Deck Loungers|Entryway Adornment Wall Displays|Premium Office Task Seating""".split("|")

D = json.load(open("h40.json"))
vocab, _ = bench.load()
gold = np.array([vocab.index(c) for c in D["g"]])
V = Vocabulary(vocab, backend="minilm")
Vt = Vocabulary(vocab, backend="tfidf")

def sc(name, texts, vv):
    hits = vv.snap(texts, k=3)
    idx = np.array([[vocab.index(l) for l, _ in r] for r in hits])
    a1 = float(np.mean([g == p[0] for p, g in zip(idx, gold)]))
    a3 = float(np.mean([g in p for p, g in zip(idx, gold)]))
    print(f"{name:44} {a1:6.3f} {a3:6.3f}")
    return a1, a3

print(f"n=40 (WANDS queries 1-40)\n{'arm':44} {'acc@1':>6} {'acc@3':>6}")
print("-" * 60)
sc("MiniLM: query -> label  [no-LLM control]",  D["q"],  V)
sc("MiniLM: Gemini flash-lite hallucination",   D["gem"], V)
sc("MiniLM: Haiku-subagent hallucination",      HAIKU,    V)
print()
sc("tfidf:  query -> label  [no-LLM control]",  D["q"],  Vt)
sc("tfidf:  Gemini flash-lite hallucination",   D["gem"], Vt)
sc("tfidf:  Haiku-subagent hallucination",      HAIKU,    Vt)
print("\nside by side:")
for q, g, gm, hk in list(zip(D["q"], D["g"], D["gem"], HAIKU))[:8]:
    print(f"  {q:34} gold={g:28} gemini={gm:28} haiku={hk}")
