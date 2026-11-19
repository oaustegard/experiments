"""Parse Tesseract's lstm_choice_mode=1 hOCR into per-timestep {char: percent} dicts."""
import html
import re
import sys


def steps(path):
    s = open(path, encoding="utf-8").read()
    out = []
    for block in re.split(r"<span class='ocrx_cinfo' id='timestep", s)[1:]:
        block = block.split("</span></span>")[0] + "</span>"
        ch = {}
        for conf, c in re.findall(r"title='x_confs (\d+)'>([^<]*)</span>", block):
            ch[html.unescape(c)] = int(conf)
        out.append(ch)
    return out


if __name__ == "__main__":
    for i, st in enumerate(steps(sys.argv[1])):
        print(i, st)
