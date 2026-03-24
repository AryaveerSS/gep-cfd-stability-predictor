import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# CONFIG

MAX_LEN = 120
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 4
BATCH_SIZE = 32
EPOCHS = 10      
LR = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD DATA

df = pd.read_csv("dataset_all_four.csv")
df["log_p"] = np.log10(df["p_initial_residual"].clip(lower=1e-12))  # BUG FIX: clip instead of add

# Combine with [SEP] boundaries — tensor names included for context
df["combined"] = (
    "(T1) " + df["string1"] + " [SEP] " +
    "(T2) " + df["string2"] + " [SEP] " +
    "(T3) " + df["string3"] + " [SEP] " +
    "(T4) " + df["string4"]
)

# BUILD VOCAB  (BUG FIX: preserve operators; correct idx start)


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

vocab = build_vocab(df["combined"])
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")
print(f"Sample vocab entries: {list(vocab.items())[:15]}")


# TOKENIZER  (BUG FIX: operators preserved; [SEP] handled)


def tokenize(text):
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

# DATASET

class GEPRunDataset(Dataset):
    def __init__(self, df):
        self.x = df["combined"].values
        self.y_label = df["label"].values
        self.y_log_p = df["log_p"].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        tokens = tokenize(self.x[idx])
        label = torch.tensor(self.y_label[idx], dtype=torch.float32)
        p_val = torch.tensor(self.y_log_p[idx], dtype=torch.float32)
        return tokens, label, p_val

train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=42)

train_dataset = GEPRunDataset(train_df)
val_dataset   = GEPRunDataset(val_df)
test_dataset  = GEPRunDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# POSITIONAL ENCODING

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])
 
# TRANSFORMER MODEL  (BUG FIX: correct order embed → PE → encoder)

class StabilityTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.positional  = PositionalEncoding(EMBED_DIM)   # moved before encoder

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=512,   # increased from 256
            dropout=0.1,
            batch_first=True,
            norm_first=True        # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)

        self.shared = nn.Sequential(
            nn.Linear(EMBED_DIM, 128),
            nn.LayerNorm(128),     # normalise before activation
            nn.GELU(),             # GELU > ReLU for transformers
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU()
        )
        self.classifier = nn.Linear(64, 1)
        self.regressor  = nn.Linear(64, 1)

    def forward(self, x):
        # Build padding mask so PAD tokens don't attend
        pad_mask = (x == 0)                     # True where padded

        x = self.embedding(x)
        x = self.positional(x)                  # BUG FIX: PE before encoder
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        cls = x[:, 0, :]                        # CLS token representation
        h   = self.shared(cls)

        return self.classifier(h).squeeze(-1), self.regressor(h).squeeze(-1)

model = StabilityTransformer().to(DEVICE)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# LOSSES + OPTIMIZER + SCHEDULER

# Compute class weight from training data
n_pos = train_df["label"].sum()
n_neg = len(train_df) - n_pos
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
print(f"pos_weight: {pos_weight.item():.2f}  (neg/pos ratio)")

bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
mse = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

ALPHA = 0.3   # regression loss weight

# TRAIN

train_acc_list, val_acc_list = [], []
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    correct = total = 0
    epoch_loss = 0.0

    for x, label, p in train_loader:
        x, label, p = x.to(DEVICE), label.to(DEVICE), p.to(DEVICE)

        optimizer.zero_grad()
        label_pred, p_pred = model(x)

        loss1 = bce(label_pred, label)
        loss2 = mse(p_pred, p)
        loss  = loss1 + ALPHA * loss2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = (torch.sigmoid(label_pred) > 0.5).float()
        correct     += (preds == label).sum().item()
        total       += label.size(0)
        epoch_loss  += loss.item()

    scheduler.step()
    train_acc = correct / total
    train_acc_list.append(train_acc)

    # --- Validation ---
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for x, label, p in val_loader:
            x, label, p = x.to(DEVICE), label.to(DEVICE), p.to(DEVICE)  # BUG FIX: p on device
            label_pred, _ = model(x)
            preds   = (torch.sigmoid(label_pred) > 0.5).float()
            correct += (preds == label).sum().item()
            total   += label.size(0)

    val_acc = correct / total
    val_acc_list.append(val_acc)

    # Save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")

    print(f"Epoch {epoch+1:02d} | Loss {epoch_loss/len(train_loader):.4f} "
          f"| Train {train_acc:.3f} | Val {val_acc:.3f} | LR {scheduler.get_last_lr()[0]:.2e}")

# TEST SET EVALUATION  (load best checkpoint)

model.load_state_dict(torch.load("best_model.pt"))
model.eval()

correct = total = 0
all_probs, all_labels = [], []

with torch.no_grad():
    for x, label, p in test_loader:
        x, label = x.to(DEVICE), label.to(DEVICE)
        label_pred, _ = model(x)
        prob  = torch.sigmoid(label_pred)
        preds = (prob > 0.5).float()
        correct     += (preds == label).sum().item()
        total       += label.size(0)
        all_probs.extend(prob.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

test_acc = correct / total
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

# Precision / Recall / F1 — better metrics for imbalanced data
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(all_labels, [1 if p > 0.5 else 0 for p in all_probs],
                             target_names=["stable", "unstable"]))
print(f"ROC-AUC: {roc_auc_score(all_labels, all_probs):.4f}")