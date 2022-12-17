from navec import Navec
import torch.nn as nn
from slovnet.model.emb import NavecEmbedding

class Extractor(nn.Module):
    def __init__(self,
                 navec_emb_path: str = 'navec_hudlit_v1_12B_500K_300d_100q.tar',
                 emb_proj_dim: int = 200,
                 hid_dim: int = 256,
                 layers: int = 2,
                 dropout: float = 0.5,
                ):
        super().__init__()
        navec = Navec.load(navec_emb_path)
        self.vocab = navec.vocab
        navec_emb_size = 300
        self.emb = NavecEmbedding(navec)
        self.emb_proj = nn.Linear(navec_emb_size, emb_proj_dim)
        self.rnn = nn.LSTM(emb_proj_dim, hid_dim, num_layers=layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.out = nn.Linear(2 * hid_dim, 1)
    
    def forward(self, inp):
        embedded = self.emb(inp)
        projected = self.emb_proj(embedded)
        out, _ = self.rnn(projected)
        out = self.out(out)
        return out
