import torch 
import torch.nn as nn
from src.positional_encoding import PositionalEncoding
from src.tree_embedding import TreeEmbedding

class StabilityTransformer(nn.Module):
    def __init__(self,vocab_size,EMBED_DIM,NUM_HEADS,NUM_LAYERS):
        super().__init__()
        self.embedding   = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.positional  = PositionalEncoding(EMBED_DIM)   # moved before encoder
        self.tree_embedding = TreeEmbedding(max_depth=20, max_size=50, d_model=EMBED_DIM) # tree embedding 

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

    def forward(self,token_ids,depth_ids,subtree_ids):
        # Build padding mask so PAD tokens don't attend
        pad_mask = (token_ids == 0)                     # True where padded

        x = self.embedding(token_ids)
        x = self.positional(x)                 # BUG FIX: PE before encoder
        tree_emb = self.tree_embedding(depth_ids,subtree_ids)
        x=x+tree_emb
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        cls = x[:, 0, :]                        # CLS token representation
        h   = self.shared(cls)

        return self.classifier(h).squeeze(-1), self.regressor(h).squeeze(-1)
