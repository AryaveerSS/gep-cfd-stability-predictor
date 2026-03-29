import torch 
import torch.nn as nn
from src.positional_encoding import PositionalEncoding
from src.tree_embedding import TreeEmbedding

class StabilityTransformer(nn.Module):
    def __init__(self,vocab_size,EMBED_DIM,NUM_HEADS,NUM_LAYERS):
        super().__init__()
        # self.embedding   = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        # self.positional  = PositionalEncoding(EMBED_DIM)   # moved before encoder
        # self.tree_embedding = TreeEmbedding(max_depth=20, max_size=50, d_model=EMBED_DIM) # tree embedding
        self.token_embed = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.depth_embed = nn.Embedding(10, EMBED_DIM) # depth 0-9
        self.subtree_embed = nn.Embedding(5, EMBED_DIM) # 5 size buckets
        self.positional = PositionalEncoding(EMBED_DIM) 

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

    def forward(self, tok_ids, dep_ids, sub_ids):
        pad_mask = (tok_ids == 0) # PAD positions

        x = self.token_embed(tok_ids) # what is this symbol
        x += self.depth_embed(dep_ids) # how deep in tree
        x += self.subtree_embed(sub_ids) # how much below
        x = self.positional(x) # sequence order

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        cls = x[:, 0, :]
        h = self.shared(cls)
        return self.classifier(h).squeeze(-1), self.regressor(h).squeeze(-1)
