"""Scene stills, two takes each, with the chosen cast portraits as references.
  python3 shots.py [ids...]  -> stills/<id>_<n>.png"""
import sys, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from img import generate

CAST = dict(V="cast/V_0.png", O="cast/O_1.png", F="cast/F_0.png", L="cast/L_0.png", D="cast/D_1.png", B="cast/B_1.png", R="cast/R_1.png", M="cast/M_0.png")
NAME = dict(V="Aurèle Monnier the old watchmaker", O="Odile Monnier", F="Félix Monnier", L="Lina Ruedi", D="Dr Vauthier", B="Bastien Girard", R="Rosa the maid", M="Inspector Matthey")
CLOCK = " Any clock face is seen at an angle or in shadow so that its hands cannot be read."

SHOTS = {
 "stairs": ("A narrow wooden staircase in a dark stone house at night; a maid in a white cap and apron climbs with a candle and a tray of champagne glasses, seen from below, her shadow huge on the wall.", "R"),
 "body": ("A watchmaker's atelier at night lit by one oil lamp: an old man lies face down on the pine floorboards beside a long workbench under tall dark windows; only his back, grey head and outstretched right arm are visible; a brass mallet lies near him; no blood, no gore. Quiet, still, tragic.", "V"),
 "regulator": ("Close view of a tall precision regulator clock in a glazed walnut case on a workshop wall at night, its long pendulum hanging motionless; the large white enamel dial fills the upper half of the frame, plain, with no hands painted on it.", ""),
 "repeater": ("Extreme close-up: an old man's lifeless hand on dark floorboards, fingers closed around an open gold pocket watch, a minute repeater, its small slide on the case edge pushed down; lamplight glints on the gold.", ""),
 "exterior": ("Night exterior, La Chaux-de-Fonds in the Swiss Jura: a long stone watchmaker's house on a snowy grid street, its top floor lit by a continuous row of long workshop windows glowing amber; a heavy man in a dark overcoat stands smoking under the front windows; hard frost, old snow, a church tower faint in the distance.", "B"),
 "arrival": ("Just after midnight, falling snow: a lean police inspector in a dark overcoat and bowler hat stands at the lit front door of a stone house, a uniformed gendarme holding a lantern beside him; fireworks faint over distant rooftops.", "M"),
 "town": ("Winter daylight panorama over La Chaux-de-Fonds: long parallel rows of stone houses on a strict grid, snow on the roofs, every top floor a band of long windows, smoke from chimneys, pine hills of the Jura beyond.", ""),
 "atelier_day": ("A watchmaker's atelier by winter daylight: a long workbench under a continuous row of tall windows, tiny tools and movements laid out on green baize, an old master watchmaker with a loupe and a young woman régleuse working side by side; a tall regulator clock on the wall.", "V L"),
 "dinner": ("Dinner by lamplight in a bourgeois Swiss dining room, 1899: at the head of the table the old watchmaker holds up a silver pocket watch and lectures; around the table his wife, a flushed young man, a young woman, a portly doctor and a big bearded merchant hold out their watches; a maid at the sideboard.", "V O F L D B R"),
 "interview": ("Small hours in a dining room: the police inspector sits at the cleared table with a notebook under a single oil lamp, an empty chair opposite him; dark windows, frost on the panes.", "M"),
 "kitchen": ("A plain stone kitchen at night: the maid stands by the range beside a long table of polished champagne glasses on a tray; a cheap American alarm clock on the shelf;" + CLOCK, "R"),
 "cards": ("A salon at night: a portly doctor with a cigar and a severe woman in black play piquet at a small card table under a lamp; a marble mantel clock behind them;" + CLOCK, "D O"),
 "piano": ("A salon at night: a woman in a black silk dress plays an upright piano by candlelight, her back straight; beside her an empty chair at a card table with cards laid down and a cigar smouldering in a dish.", "O"),
 "atelier_night": ("The atelier at night: the young régleuse with a loupe works beside the old master at the long bench under dark windows, an oil lamp between them, the tall regulator on the wall behind.", "V L"),
 "kitchen_lina": ("The kitchen at night: the young régleuse sits pale and still by the stove holding her own pocket watch, looking up at a cheap alarm clock on the shelf; the maid polishes glasses behind her;" + CLOCK, "L R"),
 "afternoon": ("Winter afternoon light in a stone kitchen: the old watchmaker, laughing, holds a cheap tin alarm clock open in his hands and turns its hands with a fingertip; nobody else in the room.", "V"),
 "library": ("A book-lined library at night: the young man in evening dress slumps in a leather armchair with a glass of brandy, his silver watch open in his palm, looking up anxiously at the ceiling.", "F"),
 "ceiling": ("Looking up at the pine-board ceiling of a dark library, a crack of lamplight between the boards from the room above; a chandelier hook; dust drifting in the light. Ominous, quiet.", ""),
 "door": ("A closed panelled library door at night seen from inside the room; under the door a thin line of light from the stairwell is broken by the shadow of feet passing.", ""),
 "street": ("Night street in deep frost: a big bearded man in a heavy overcoat smokes a thin cigar under the lit ground-floor windows of a stone house, a warm glow from the salon window beside him; above, the long top-floor workshop windows glow amber, one opened a crack.", "B"),
 "hall": ("A dim hall with a staircase in an old stone house: a tall painted Morbier long-case clock with a swinging brass pendulum; a big bearded man in an overcoat with snow on his shoulders comes in through the front door;" + CLOCK, "B"),
 "interview_empty": ("The dining room before dawn: an empty table, a notebook and a pencil, a guttered candle, the lamp turned low, six empty chairs pushed back.", ""),
 "gathering": ("Dawn in a bourgeois salon, Poirot-like drawing-room gathering: the police inspector stands by the mantelpiece; seated around the room are a severe widow in black, a dishevelled young man, a young woman in grey, a portly doctor lighting a cigar, a big bearded merchant and a maid in an apron standing by the door; grey morning light at the windows.", "M O F L D B R"),
 "morbier": ("A tall old Morbier long-case clock in a dim hall at dawn, its painted floral case and long brass pendulum, the police inspector standing beside it looking up at it thoughtfully;" + CLOCK, "M"),
 "kitchen_clock": ("Still life in grey dawn light: a cheap American tin alarm clock on a kitchen shelf beside copper pans and a row of champagne glasses;" + CLOCK, ""),
 "climb": ("Reconstruction: seen from the top of a dark stairwell, a portly man in a black frock coat climbs quickly and quietly; below him warm light spills from an open salon door; his face in shadow.", "D"),
 "pendulum": ("Reconstruction: in a dark atelier a portly man's hand in a black sleeve reaches into the glazed case of a tall regulator clock and stills its brass pendulum; his face out of frame; lamplight.", ""),
 "accused": ("Close portrait at dawn: the portly silver-haired doctor in a black frock coat stubs out his cigar in a dish, his face resigned and grey, eyes lowered; the window light behind him.", "D"),
 "letter": ("Dawn: a writing desk by a frosted window, a sealed letter addressed in copperplate lying beside a pen, a gold pince-nez and a cigar case; the room empty.", ""),
 "dawn": ("New Year's morning 1900 over La Chaux-de-Fonds: pale gold sunrise over snowy rooftops and long rows of workshop windows, a church tower, smoke rising, very still.", ""),
}

def job(sid, n):
    out = f"stills/{sid}_{n}.png"
    if Path(out).exists(): return out
    prompt, who = SHOTS[sid]
    refs = [CAST[k] for k in who.split()]
    if refs:
        prompt += " Characters in this scene: " + ", ".join(NAME[k] for k in who.split()) + " (see the references)."
    return generate(prompt, out, refs=refs, temperature=0.85)

if __name__ == "__main__":
    ids = sys.argv[1:] or list(SHOTS)
    jobs = [(s, n) for s in ids for n in range(2)]
    with ThreadPoolExecutor(10) as ex:
        for r in ex.map(lambda a: job(*a), jobs): print(r, flush=True)
