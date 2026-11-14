"""Plain Okapi BM25 over the corpus paragraphs (title + text), k1 1.2, b 0.75."""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

TOKEN = re.compile(r"\w+")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    def __init__(self, path: Path, k1: float = 1.2, b: float = 0.75):
        self.docs = [json.loads(l) for l in path.open()]
        self.k1, self.b = k1, b
        self.post = defaultdict(list)
        self.len = []
        for d in self.docs:
            tf = Counter(tokens(d["title"] + " " + d["text"]))
            self.len.append(sum(tf.values()))
            for t, c in tf.items():
                self.post[t].append((d["id"], c))
        n = len(self.docs)
        self.avg = sum(self.len) / n
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def search(self, query: str, k: int = 5) -> list[dict]:
        score = defaultdict(float)
        for t in set(tokens(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, c in self.post[t]:
                norm = c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
                score[i] += idf * norm
        top = sorted(score, key=lambda i: (-score[i], i))[:k]
        return [self.docs[i] for i in top]


def render(hits: list[dict]) -> str:
    return "\n".join(f"[{j + 1}] {h['title']}: {h['text']}" for j, h in enumerate(hits))
