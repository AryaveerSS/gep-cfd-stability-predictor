import torch
import torch.nn as nn

class TreeEmbedding(nn.Module):
    def __init__(self, max_depth=20, max_size=50, d_model=128):
        super().__init__()
        
        self.depth_embedding = nn.Embedding(max_depth, d_model)
        self.subtree_embedding = nn.Embedding(max_size, d_model)

    def forward(self, depth_ids, subtree_ids):
        """
        depth_ids:   (batch_size, seq_len)
        subtree_ids: (batch_size, seq_len)
        """

        depth_emb = self.depth_embedding(depth_ids)
        subtree_emb = self.subtree_embedding(subtree_ids)

        return depth_emb + subtree_emb