"""Small, inspectable BM25 corpus. Scores are relevance, never confidence."""
import math
import re
from collections import Counter

STOP = set('a an the is are was were what who when where how which of in on for to and by did does do has have it this that with as'.split())


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def chunks(text, size=110, overlap=20):
    words = text.split()
    for start in range(0, len(words), size - overlap):
        yield ' '.join(words[start:start + size])
        if start + size >= len(words):
            break


class BM25:
    def __init__(self, documents):
        self.entries = []
        for doc in documents:
            for i, chunk in enumerate(chunks(doc['text'])):
                self.entries.append({**doc, 'text': chunk, 'chunk': i})
        self.counts = [Counter(tokens(d['title'] + ' ' + d['text'])) for d in self.entries]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avg = sum(self.lengths) / max(1, len(self.lengths)) or 1
        self.df = Counter(w for c in self.counts for w in c)

    def search(self, query, k=3):
        terms = set(tokens(query))
        scored = []
        for doc, counts, length in zip(self.entries, self.counts, self.lengths):
            score = 0.0
            for term in terms:
                freq = counts[term]
                if freq:
                    idf = math.log(1 + (len(self.entries) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    score += idf * freq * 2.5 / (freq + 1.5 * (0.25 + 0.75 * length / self.avg))
            coverage = len(terms.intersection(counts)) / max(1, len(terms))
            if score > 0 and coverage >= 0.6:
                scored.append({**doc, 'score': round(score, 4), 'query_coverage': round(coverage, 3)})
        return sorted(scored, key=lambda d: (-d['score'], d['id'], d['chunk']))[:k]


def extract(query, docs):
    """Return a verbatim passage. This baseline makes no generative claim."""
    if not docs:
        return None
    terms = set(tokens(query))
    sentences = re.split(r'(?<=[.!?])\s+', docs[0]['text'])
    return max(sentences, key=lambda s: len(terms.intersection(tokens(s))))

