import torch 
import torch.nn as nn
from src.positional_encoding import PositionalEncoding

class StabilityTransformer(nn.Module):
    def __init__(self,vocab_size,EMBED_DIM,NUM_HEADS,NUM_LAYERS):
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
