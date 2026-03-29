import torch 

import torch
import math
import pandas as pd 
# -----------------------------
# Basic tokenization
# -----------------------------
# def tokenize_raw(text):
#     """Split prefix expression into tokens, preserving all operators."""
#     text = text.replace(",", " ")
#     return [t for t in text.split() if t]
import numpy as np
import math

def bucket_constant(val_str):
    v = abs(float(val_str))
    if v == 0.0: return "CONST_ZERO"
    if v < 0.01: return "CONST_TINY"
    if v < 0.1: return "CONST_SMALL"
    if v < 1.0: return "CONST_MED"
    if v < 5.0: return "CONST_LARGE"
    return "CONST_HUGE"

def tokenize_raw(text):
    """Split prefix string and bucket numeric constants."""
    out = []
    for t in text.split(): # prefix already space-separated
        if t == "[SEP]":
            out.append("[SEP]")
        else:
            try: out.append(bucket_constant(t))
            except: out.append(t) # +, -, *, I1, I2
    return out

OPERATORS = {'+', '-', '*'}

def parse_prefix(tokens, idx, depth, depths, sizes):
    """Walk prefix token list, fill depths[] and sizes[]."""
    if idx >= len(tokens) or tokens[idx] == "[SEP]":
        return idx, 0
    depths[idx] = depth
    if tokens[idx] in OPERATORS:
        idx, ls = parse_prefix(tokens, idx+1, depth+1, depths, sizes)
        idx, rs = parse_prefix(tokens, idx, depth+1, depths, sizes)
        sizes[idx-ls-rs-1] = 1 + ls + rs # ← correct: set by original idx
        return idx, 1 + ls + rs
    else:
        sizes[idx] = 1
        return idx+1, 1

def get_subtree_id(size):
    if size <= 1: return 0 # leaf
    if size <= 3: return 1 # tiny
    if size <= 7: return 2 # small
    if size <= 15: return 3 # medium
    return 4 # large


# new build vocab 
def build_vocab(strings):
    vocab = {"[PAD]":0, "[CLS]":1, "[SEP]":2}
    idx = 3
    for s in strings:
        for t in tokenize_raw(s):
            if t not in vocab:
                vocab[t] = idx; idx += 1
    return vocab

# Final vocab ≈ 14 tokens only:
# [PAD],[CLS],[SEP],+,-,*,I1,I2,
# CONST_ZERO,CONST_TINY,CONST_SMALL,
# CONST_MED,CONST_LARGE,CONST_HUGE

# new tokenize 

def tokenize(text,vocab,MAX_LEN):
    raw = ["[CLS]"] + tokenize_raw(text)
    n = len(raw)

    depths = [0] * n
    sizes = [0] * n

    # parse each of the 4 expressions (split by [SEP])
    i = 1 # skip [CLS]
    while i < n:
        if raw[i] == "[SEP]":
            depths[i] = 0; sizes[i] = 0
            i += 1
        else:
            i, _ = parse_prefix(raw, i, 0, depths, sizes)

    def pad(lst):
        lst = lst[:MAX_LEN]
        lst += [0] * (MAX_LEN - len(lst))
        return torch.tensor(lst, dtype=torch.long)

    token_ids = [vocab.get(t, 0) for t in raw]
    depth_ids = [min(d, 9) for d in depths]
    subtree_ids= [get_subtree_id(s) for s in sizes]

    return pad(token_ids), pad(depth_ids), pad(subtree_ids)