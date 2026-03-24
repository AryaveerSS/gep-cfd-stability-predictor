import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from src.config import *
from src.tokenizer import build_vocab
from src.dataset import GEPRunDataset
from src.model import StabilityTransformer
from src.train import train_model
from src.evall import evaluate

# Load data
df = pd.read_csv("data/dataset_all_four.csv")

df["log_p"] = np.log10(df["p_initial_residual"].clip(lower=1e-12))

df["combined"] = (
    "(T1) " + df["string1"] + " [SEP] " +
    "(T2) " + df["string2"] + " [SEP] " +
    "(T3) " + df["string3"] + " [SEP] " +
    "(T4) " + df["string4"]
)

# vocab
vocab = build_vocab(df["combined"])
vocab_size = len(vocab)

# split
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

# datasets
train_dataset = GEPRunDataset(train_df, vocab, MAX_LEN)
val_dataset = GEPRunDataset(val_df, vocab, MAX_LEN)
test_dataset = GEPRunDataset(test_df, vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# model
model = StabilityTransformer(vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS).to(DEVICE)

# loss
n_pos = train_df["label"].sum()
n_neg = len(train_df) - n_pos
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)

bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
mse = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# train
train_model(model, train_loader, val_loader, optimizer, scheduler, bce, mse, DEVICE, ALPHA, EPOCHS)

# evaluate
model.load_state_dict(torch.load("best_model.pt"))
evaluate(model, test_loader, DEVICE)