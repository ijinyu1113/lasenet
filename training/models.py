import torch
import torch.nn as nn
import math

class LaseNetRNN(nn.Module):
    """
    PyTorch implementation of the original LaseNet RNN (bi-GRU) model.
    """
    def __init__(self, feature_dim, continuous_output_dim, units=256, dropout=0.2, dropout1=0.1, dropout2=0.05):
        super(LaseNetRNN, self).__init__()
        self.encoder = nn.GRU(
            input_size=feature_dim,
            hidden_size=units,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.dense1 = nn.Linear(units * 2, int(units / 2)) # units * 2 because bidirectional
        self.dropout1 = nn.Dropout(dropout1)
        self.dense2 = nn.Linear(int(units / 2), int(units / 4))
        self.dropout2 = nn.Dropout(dropout2)
        self.output_layer = nn.Linear(int(units / 4), continuous_output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        encoder_outputs, _ = self.encoder(x)
        x = self.dropout(encoder_outputs)
        x = self.relu(self.dense1(x))
        x = self.dropout1(x)
        x = self.relu(self.dense2(x))
        x = self.dropout2(x)
        continuous_latent = self.output_layer(x)
        return continuous_latent

class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding for Transformer models.
    """
    def __init__(self, d_model, max_len=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class LaseNetTransformer(nn.Module):
    """
    A Transformer-based version of LaseNet for comparison.
    """
    def __init__(self, feature_dim, continuous_output_dim, nhead=8, num_encoder_layers=3, dim_feedforward=256, dropout=0.25):
        super(LaseNetTransformer, self).__init__()
        self.d_model = nhead * 32 # Embedding dimension
        self.input_embed = nn.Linear(feature_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.output_layer = nn.Linear(self.d_model, continuous_output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.input_embed(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.dropout(output)
        output = self.output_layer(output)
        return output
