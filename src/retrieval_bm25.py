# src/retrieval_bm25.py
from rank_bm25 import BM25Okapi
import pandas as pd
import re
import json

def tokenize(text):
    if not isinstance(text, str): return []
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return text.split()

def build_bm25(corpus_texts):
    tokenized = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

def get_topk(bm25, corpus_texts, query, k=10):
    qtok = tokenize(query)
    scores = bm25.get_scores(qtok)
    topk_idx = list(sorted(range(len(scores)), key=lambda i: -scores[i]))[:k]
    return [(idx, corpus_texts[idx], scores[idx]) for idx in topk_idx]
