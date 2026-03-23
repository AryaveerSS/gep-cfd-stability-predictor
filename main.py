import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# --------------------------
# CONFIG
# --------------------------

MAX_LEN = 120
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 4
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# LOAD DATA
# --------------------------

df = pd.read_csv("dataset_all_four.csv")

# log transform p value
df["log_p"] = np.log10(df["p_initial_residual"] + 1e-12)

# combine 4 strings using separators
df["combined"] = (
    df["string1"] + " [SEP] " +
    df["string2"] + " [SEP] " +
    df["string3"] + " [SEP] " +
    df["string4"]
)

# --------------------------
# BUILD VOCAB
# --------------------------

def build_vocab(strings):
    vocab = {"[PAD]":0, "[CLS]":1,"[SEP]":2}
    idx = 2

    for s in strings:
        tokens = s.replace("(", "").replace(")", "").replace("+"," , ").replace("*"," , ").split(",")
        for t in tokens:
            t = t.strip()
            if t and t not in vocab:
                vocab[t] = idx
                idx += 1

    return vocab

vocab = build_vocab(df["combined"])
vocab_size = len(vocab)

# --------------------------
# TOKENIZER
# --------------------------

def tokenize(text):

    tokens = text.replace("(", "").replace(")", "").replace("+"," , ").replace("*"," , ").split(",")

    ids = [vocab["[CLS]"]]

    for t in tokens:
        t = t.strip()
        if t:
            ids.append(vocab.get(t,0))

    if len(ids) < MAX_LEN:
        ids += [0]*(MAX_LEN-len(ids))
    else:
        ids = ids[:MAX_LEN]

    return torch.tensor(ids)

# --------------------------
# DATASET
# --------------------------

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

# # first split train vs temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

# split temp into validation + test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)
# train_dataset = GEPRunDataset(train_df)
# val_dataset = GEPRunDataset(val_df)

train_dataset = GEPRunDataset(train_df)
val_dataset   = GEPRunDataset(val_df)
test_dataset  = GEPRunDataset(test_df)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# seperate positional encoding

import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=200):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]

        return x

# --------------------------
# TRANSFORMER MODEL
# --------------------------

class StabilityTransformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMBED_DIM)
        self.positional = PositionalEncoding(EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )

        self.shared = nn.Sequential(
            nn.Linear(EMBED_DIM,128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.classifier = nn.Linear(64,1)
        self.regressor = nn.Linear(64,1)

    def forward(self,x):

        x = self.embedding(x)
        x = self.encoder(x)
        x = self.positional(x)
        cls = x[:,0,:]

        h = self.shared(cls)

        label_out = self.classifier(h)
        p_out = self.regressor(h)

        return label_out.squeeze(), p_out.squeeze()

model = StabilityTransformer().to(DEVICE)

# --------------------------
# LOSSES
# --------------------------

pos_weight = torch.tensor([2.0]).to(DEVICE)
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
mse = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --------------------------
# TRAIN
# --------------------------

train_acc_list = []
val_acc_list = []

for epoch in range(EPOCHS):

    model.train()

    correct = 0
    total = 0

    for x,label,p in train_loader:

        x,label,p = x.to(DEVICE), label.to(DEVICE), p.to(DEVICE)

        optimizer.zero_grad()

        label_pred,p_pred = model(x)

        loss1 = bce(label_pred,label)
        loss2 = mse(p_pred,p)

        loss = loss1 + 0.3*loss2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = (torch.sigmoid(label_pred)>0.5).float()

        correct += (preds==label).sum().item()
        total += label.size(0)

    train_acc = correct/total
    train_acc_list.append(train_acc)

    # validation

    model.eval()
    correct=0
    total=0

    with torch.no_grad():

        for x,label,p in val_loader:

            x,label = x.to(DEVICE), label.to(DEVICE)

            label_pred,_ = model(x)

            preds = (torch.sigmoid(label_pred)>0.5).float()

            correct += (preds==label).sum().item()
            total += label.size(0)

    val_acc = correct/total
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1} | Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}")

# --------------------------
# TEST SET EVALUATION
# --------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for x, label, p in test_loader:

        x = x.to(DEVICE)
        label = label.to(DEVICE)

        label_pred, p_pred = model(x)

        preds = (torch.sigmoid(label_pred) > 0.5).float()

        correct += (preds == label).sum().item()
        total += label.size(0)

test_acc = correct / total

import random

def check_random_predictions(model, test_df, n=10):

    model.eval()

    samples = test_df.sample(n)

    print("\nRandom Test Predictions:\n")

    with torch.no_grad():

        for _, row in samples.iterrows():

            text = row["combined"]

            x = tokenize(text).unsqueeze(0).to(DEVICE)

            label_pred, p_pred = model(x)

            prob = torch.sigmoid(label_pred).item()

            pred_label = 1 if prob > 0.5 else 0

            print("Run:", row["run"])
            print("True Label:", row["label"])
            print("Pred Label:", pred_label)

            print("True p:", row["log_p"])
            pred_log_p = p_pred.item()
            pred_p = 10 ** pred_log_p

            print("Pred log p:", pred_log_p)
            print("Pred p:", pred_p)

            print("--------------")


print("Final Test Accuracy:", test_acc)

check_random_predictions(model, test_df, n=10)
# --------------------------
# PLOT
# --------------------------
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.plot(train_acc_list,label="Train Accuracy")
plt.plot(val_acc_list,label="Validation Accuracy")

plt.axhline(y=test_acc,color='red',linestyle='--',label="Test Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Performance")

plt.legend()
plt.show()