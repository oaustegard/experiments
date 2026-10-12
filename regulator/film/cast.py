"""Character reference portraits, two takes each: python3 cast.py -> cast/<k>_<n>.png"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from img import generate

LOOK = {
 "V": "Aurèle Monnier, master watchmaker, 62: lean, stooped, close-cropped grey hair, short white beard, wire spectacles pushed up on his forehead, a jeweller's loupe on a cord, dark waistcoat with shirtsleeves rolled, ink-stained fingers, a sharp amused mouth",
 "O": "Odile Monnier, 55: tall, upright, grey-streaked dark hair in a severe chignon, pale composed face, high-necked black silk evening dress with a small cameo brooch, cool grey eyes, no smile",
 "F": "Félix Monnier, 28: handsome, slightly dishevelled, dark wavy hair falling over the brow, thin moustache, loosened white tie, evening tailcoat, a tarnished silver pocket watch on a chain, anxious charming eyes, a little flushed from brandy",
 "L": "Lina Ruedi, 24, régleuse: slight, very straight posture, fair hair pinned tightly back, plain dark grey wool dress with a white collar, a watchmaker's loupe hanging at her neck, serious level gaze, freckles, capable hands",
 "D": "Dr Casimir Vauthier, physician, 58: portly and genial, silver hair brushed back, neat silver mutton-chop whiskers, gold pince-nez, black frock coat and waistcoat with a gold watch chain, a cigar between two fingers, warm heavy-lidded eyes",
 "B": "Bastien Girard, établisseur, 45: big broad-shouldered man, ruddy weathered face, short brown beard, heavy dark overcoat over a plain suit, a cheap thin cigar, blunt impatient expression",
 "R": "Rosa Häfeli, maid and cook, 40: sturdy, round rosy face, brown hair under a white cap, white apron over a dark blue dress, sleeves pushed up, frightened but proud expression",
 "M": "Inspector Léon Matthey of the cantonal police, 41: lean, clean-shaven except a short dark moustache, thoughtful grey eyes, dark overcoat with snow on the shoulders, bowler hat in hand, a small notebook, patient expression",
}

def job(k, n):
    out = f"cast/{k}_{n}.png"
    if Path(out).exists(): return out
    return generate(f"Half-length portrait, three-quarter view, plain dark interior background lit by a single lamp: {LOOK[k]}. The year is 1899.", out, temperature=0.9)

if __name__ == "__main__":
    jobs = [(k, n) for k in LOOK for n in range(2)]
    with ThreadPoolExecutor(8) as ex:
        for r in ex.map(lambda a: job(*a), jobs): print(r, flush=True)
