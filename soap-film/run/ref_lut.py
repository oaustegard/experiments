"""Independent reference for the page's thin-film colour table.

The page sums an Airy formula every 2.5 nm against the CIE 1931 observer and D65 (5 nm tables,
interpolated), with a two-term Cauchy fit for water's index. This script does the same job by a
different route: the characteristic-matrix method for one layer, colour-science's own 1 nm tables,
and either a constant index (1.333) or the page's dispersion formula. It writes linear sRGB reflectance (D65 white = 1,1,1, not gamut-mapped) to
data/ref_lut.json for run/check.mjs to compare against.

    python3 run/ref_lut.py
"""
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import colour

lam = np.arange(380.0, 781.0, 1.0)
shape = colour.SpectralShape(380, 780, 1)
cmf = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(shape).values      # (401, 3)
d65 = colour.SDS_ILLUMINANTS["D65"].copy().align(shape).values                               # (401,)
M = colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB
white = (d65[:, None] * cmf).sum(0)


def reflectance(d, cosi, n):
    """|r|^2 of an air / film / air stack, s and p averaged, by the characteristic matrix of the layer."""
    sint = np.sqrt(1 - cosi**2) / n
    cost = np.sqrt(1 - sint**2)
    phi = 2 * np.pi * n * d * cost / lam
    out = 0.0
    for eta0, eta in ((cosi, n * cost), (1 / cosi, n / cost)):       # s, then p
        m11 = np.cos(phi); m12 = -1j * np.sin(phi) / eta; m21 = -1j * eta * np.sin(phi); m22 = np.cos(phi)
        b = m11 + m12 * eta0; c = m21 + m22 * eta0
        r = (eta0 * b - c) / (eta0 * b + c)
        out = out + 0.5 * np.abs(r) ** 2
    return out


def rgb(d, cosi, n):
    xyz = (reflectance(d, cosi, n)[:, None] * d65[:, None] * cmf).sum(0) / white[1]
    return (M @ xyz) / (M @ (white / white[1]))


thick = list(range(0, 1601, 10)) + list(range(1650, 3001, 50))
angles = [1.0, 0.9, 0.7]
cauchy = 1.324 + 3046.0 / lam**2
ref = {"thickness_nm": thick, "cos_incidence": angles,
       "constant_index": [[rgb(d, c, 1.333).tolist() for d in thick] for c in angles],
       "page_dispersion": [[rgb(d, c, cauchy).tolist() for d in thick] for c in angles]}
out = Path(__file__).resolve().parents[1] / "data" / "ref_lut.json"
out.write_text(json.dumps(ref))
print("wrote", out, "peak reflectance (luminance) at normal incidence:",
      round(max(0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in ref["constant_index"][0]), 4))
