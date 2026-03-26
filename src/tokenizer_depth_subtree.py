import torch 

import torch
import math

# -----------------------------
# Basic tokenization
# -----------------------------
def tokenize_raw(text):
    """Split prefix expression into tokens, preserving all operators."""
    text = text.replace(",", " ")
    return [t for t in text.split() if t]


# -----------------------------
# Build vocabulary
# -----------------------------
def build_vocab(strings):
    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2}
    idx = 3

    for s in strings:
        for t in tokenize_raw(s):
            if t == "[SEP]":
                continue
            if t not in vocab:
                vocab[t] = idx
                idx += 1
    return vocab


# -----------------------------
# Depth computation
# -----------------------------
def compute_depths(tokens):
    stack = []
    depths = []
    operators = {'+', '-', '*'}

    for token in tokens:
        if not stack:
            depth = 0
        else:
            depth = stack[-1][1] + 1

        depths.append(depth)

        if token in operators:
            stack.append([2, depth])

        while stack:
            stack[-1][0] -= 1
            if stack[-1][0] == 0:
                stack.pop()
            else:
                break

    return depths


# -----------------------------
# Subtree size computation
# -----------------------------
# FIX THIS FUNCTION
def compute_subtree_sizes(tokens):
    n = len(tokens)
    sizes = [0] * n
    stack = []
    operators = {'+', '-', '*'}

    for i in reversed(range(n)):
        token = tokens[i]

        if token not in operators:
            size = 1
        else:
            # SAFE CHECK (critical fix)
            if len(stack) >= 2:
                left = stack.pop()
                right = stack.pop()
                size = 1 + left + right
            else:
                size = 1  # fallback for invalid expressions

        size = int(math.log2(size)) if size > 0 else 0
        sizes[i] = size
        stack.append(size if size > 0 else 1)

    return sizes


# -----------------------------
# Main tokenize function
# -----------------------------
def tokenize(text, vocab, MAX_LEN):
    tokens = tokenize_raw(text)

    # -----------------------
    # Token IDs
    # -----------------------
    ids = [vocab["[CLS]"]]

    for t in tokens:
        if t == "[SEP]":
            ids.append(vocab["[SEP]"])
        else:
            ids.append(vocab.get(t, 0))  # unknown → PAD

    # -----------------------
    # Tree features
    # -----------------------
    depth_ids = compute_depths(tokens)
    subtree_ids = compute_subtree_sizes(tokens)

    # Align with CLS token
    depth_ids = [0] + depth_ids
    subtree_ids = [0] + subtree_ids

    # -----------------------
    # Padding / Truncation
    # -----------------------
    if len(ids) < MAX_LEN:
        pad_len = MAX_LEN - len(ids)

        ids += [0] * pad_len
        depth_ids += [0] * pad_len
        subtree_ids += [0] * pad_len

    else:
        ids = ids[:MAX_LEN]
        depth_ids = depth_ids[:MAX_LEN]
        subtree_ids = subtree_ids[:MAX_LEN]

    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(depth_ids, dtype=torch.long),
        torch.tensor(subtree_ids, dtype=torch.long)
    )