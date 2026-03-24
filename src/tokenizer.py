import torch 

def tokenize_raw(text):
    """Split prefix expression into tokens, preserving all operators."""
    # Replace commas with spaces, then split on whitespace
    # This preserves +, -, * as standalone tokens
    text = text.replace(",", " ")
    return [t for t in text.split() if t]

def build_vocab(strings):
    # BUG FIX: idx starts at 3 so [SEP]=2 is not overwritten
    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2}
    idx = 3

    for s in strings:
        for t in tokenize_raw(s):
            if t == "[SEP]":
                continue  # already in vocab
            if t not in vocab:
                vocab[t] = idx
                idx += 1
    return vocab

def tokenize(text,vocab,MAX_LEN):
    ids = [vocab["[CLS]"]]
    for t in tokenize_raw(text):    
        if t == "[SEP]":
            ids.append(vocab["[SEP]"])
        else:
            ids.append(vocab.get(t, 0))  # 0 = [PAD] for unseen tokens
    # Pad or truncate
    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]
    return torch.tensor(ids, dtype=torch.long)
