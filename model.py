import torch
import torch.nn as nn

import config


class ModalBranch(nn.Module):
    """Single-modality feature extractor: BiGRU -> FC -> ReLU -> Dropout -> FC -> ReLU -> Dropout."""

    def __init__(self, input_dim, hidden_dim, fc1_dim, fc2_dim, dropout):
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(2 * hidden_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, h_n = self.bigru(x)  # h_n: (2, batch, hidden)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)  # (batch, 2*hidden)
        h = self.dropout(self.relu(self.fc1(h)))
        h = self.dropout(self.relu(self.fc2(h)))
        return h  # (batch, fc2_dim)


class AttentionFusion(nn.Module):
    """
    Attention-based feature-level fusion (Fig. 5).
    Uses tanh-based additive attention to compute importance weights
    for each modality, then produces a weighted combination.
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.W = nn.Linear(feature_dim, feature_dim)
        self.v = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, feat_disp, feat_acc):
        stacked = torch.stack([feat_disp, feat_acc], dim=1)  # (batch, 2, dim)
        scores = self.v(torch.tanh(self.W(stacked)))         # (batch, 2, 1)
        weights = torch.softmax(scores, dim=1)               # (batch, 2, 1)
        fused = (weights * stacked).sum(dim=1)               # (batch, dim)
        return fused


class AMFBiGRU(nn.Module):
    """
    Attention-based Multi-modal Fusion Bidirectional GRU (AMF-BiGRU).
    Paper: Liu et al., Engineering Applications of AI, 2026.
    """

    def __init__(
        self,
        disp_input_dim=config.DISP_FEATURES,
        acc_input_dim=config.ACC_FEATURES,
        hidden_dim=config.BIGRU_HIDDEN,
        fc1_dim=config.FC1_DIM,
        fc2_dim=config.FC2_DIM,
        output_dim=config.OUTPUT_DIM,
        dropout=config.DROPOUT,
    ):
        super().__init__()
        self.disp_branch = ModalBranch(disp_input_dim, hidden_dim, fc1_dim, fc2_dim, dropout)
        self.acc_branch = ModalBranch(acc_input_dim, hidden_dim, fc1_dim, fc2_dim, dropout)
        self.attention_fusion = AttentionFusion(fc2_dim)
        self.output_layer = nn.Linear(fc2_dim, output_dim)

    def forward(self, x_disp, x_acc):
        feat_disp = self.disp_branch(x_disp)
        feat_acc = self.acc_branch(x_acc)
        fused = self.attention_fusion(feat_disp, feat_acc)
        out = self.output_layer(fused)
        return out
