"""Who is on the ride. Same cast and colours as v2; Karim and Angelo have no beards (Oskar's correction)."""
import numpy as np

CAST = {
    'me':     dict(id='me', name='ME', jersey='#1d6fb8', skin='#e8b894', helm='#f4f4f4', bike='gravel', frame='#6b7f3a', build=1.04, hair='#5a3b24', stache=True, face=True, glasses=False),
    'angelo': dict(id='angelo', name='ANGELO', jersey='#2a9d8f', skin='#c98f63', helm='#262626', bike='single', frame='#cfd4dc', build=0.96, hair='#1b1b1b', pack=True, face=True, glasses=False),
    'gary':   dict(id='gary', name='GARY', jersey='#f4a261', skin='#f0c4a0', helm='#e63946', bike='road', frame='#f4a261', build=1.0, hair='#8a8a8a', stache=True, upright=1.0, face=True, glasses=False),
    'phil':   dict(id='phil', name='PHIL', jersey='#e63946', skin='#e0ac86', helm='#ffffff', bike='road', frame='#e63946', build=1.05, hair='#3a2a1a', face=True, glasses=True),
    'hulk':   dict(id='hulk', name='THE HULK', jersey='#3d3d4a', skin='#d9a07a', helm='#ffb000', bike='road', frame='#2b2b33', build=1.22, hair='#1b1b1b', stripe='#ffb000', face=True, glasses=False),
    'karim':  dict(id='karim', name='KARIM', jersey='#ffd23f', skin='#b98056', helm='#1b1b1b', bike='road', frame='#ffd23f', build=1.0, hair='#1b1b1b', face=True, glasses=True),
}
NAMED = ['karim', 'phil', 'hulk', 'gary', 'angelo', 'me']

_FAR_J = ['#8e44ad', '#16a085', '#d35400', '#2c3e50', '#c0392b', '#f1c40f', '#27ae60', '#e84393', '#0984e3', '#7f8c8d', '#e67e22', '#2980b9']
_SKINS = ['#e8b894', '#c98f63', '#8d5a3b', '#f0c4a0']
_HELMS = ['#ffffff', '#222222', '#e63946', '#ffd23f']


def far_specs(n=32):
    rng = np.random.default_rng(11)
    out = []
    for i in range(n):
        j = _FAR_J[i % len(_FAR_J)]
        out.append(dict(id=f'far{i}', name=f'FAR{i}', jersey=j, skin=_SKINS[i % 4], helm=_HELMS[(i // 2) % 4], bike='road', frame=j,
                        build=float(0.94 + 0.12 * rng.random()), hair='#3a2a1a', glasses=bool(i % 3), face=False))
    return out
