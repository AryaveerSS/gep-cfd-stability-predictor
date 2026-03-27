import torch

MAX_LEN = 200
EMBED_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 4
BATCH_SIZE = 32
EPOCHS = 10      
LR = 3e-4
ALPHA=0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")