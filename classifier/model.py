import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch.conv.gatconv import GATConv
import pandas as pd
from sklearn.preprocessing import normalize

from sklearn.metrics import average_precision_score, f1_score, zero_one_loss,hamming_loss,accuracy_score,multilabel_confusion_matrix
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import coverage_error

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(GATConv(in_feats=in_dim, out_feats=out_dim, num_heads=1))

    def forward(self, g, h):
        head_outs = [attn_head(g, h).squeeze(1) for attn_head in self.heads]
        return torch.cat(head_outs, dim=1)


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout_rate=0.1):
        super(GraphTransformerLayer, self).__init__()
        self.multi_head_gat = MultiHeadGATLayer(in_dim, out_dim // num_heads, num_heads)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, g, h):
        h_in = h
        h = self.multi_head_gat(g, h)
        h = self.dropout(h)
        h = self.norm(h + h_in)

        h_in = h
        h = self.feed_forward(h)
        h = self.dropout(h)
        h = self.norm(h + h_in)
        return h

class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, num_layers):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(GraphTransformerLayer(
                in_dim if i == 0 else hidden_dim,
                hidden_dim,
                num_heads,
                dropout_rate=0.1
            ))
        self.out_projection = nn.Linear(hidden_dim, out_dim)

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h)
        return self.out_projection(h)




