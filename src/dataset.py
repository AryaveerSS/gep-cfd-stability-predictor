import torch
from torch.utils.data import Dataset
from src.tokenizer_depth_subtree import tokenize


class GEPRunDataset(Dataset):
    def __init__(self, df,vocab, MAX_LEN):
        self.x = df["combined"].values
        self.y_label = df["label"].values
        self.y_log_p = df["log_p"].values
        self.vocab=vocab
        self.MAX_LEN=MAX_LEN

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # tokens = tokenize(self.x[idx],self.vocab, self.MAX_LEN)
        token_ids, depth_ids, subtree_ids = tokenize(self.x[idx],self.vocab, self.MAX_LEN)

        label = torch.tensor(self.y_label[idx], dtype=torch.float32)
        p_val = torch.tensor(self.y_log_p[idx], dtype=torch.float32)
        return token_ids,depth_ids,subtree_ids,label, p_val