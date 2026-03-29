import pandas as pd
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
df = pd.read_csv("data/dataset_all_four.csv")
df["combined"] = (
    "(T1) " + df["string1"] + " [SEP] " +
    "(T2) " + df["string2"] + " [SEP] " +
    "(T3) " + df["string3"] + " [SEP] " +
    "(T4) " + df["string4"]
)
_raw_lens = df["combined"].apply(
    lambda s: len(tokenize_raw(s)))
MAX_LEN = int(_raw_lens.max()) + 20
print(f"MAX_LEN set to {MAX_LEN}")