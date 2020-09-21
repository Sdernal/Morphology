import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SimpleTagger(nn.Module):
    def __init__(self, output_dim: int, embedding_matrix: torch.FloatTensor, hidden_dim=300,
                 feedforward_dim=100, dropout_rate=0.1):
        super(SimpleTagger, self).__init__()
        embedding_dim = embedding_matrix.size(-1)
        self.embedder = nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=0)
        self.encoder = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=2, bidirectional=True,
                              batch_first=True)
        self.feedforward = nn.Linear(2*hidden_dim, feedforward_dim)
        self.out = nn.Linear(feedforward_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.LongTensor):
        # x: (batch_size, seq_len)
        mask = (x != 0).to(torch.long)
        x = self.embedder(x)
        # x: (batch_size, seq_len, embedding_dim)
        lengths = mask.sum(dim=1)
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.encoder(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        x = self.feedforward(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.out(x)
