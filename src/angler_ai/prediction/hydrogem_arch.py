"""HydroGEM model architecture vendored from the published HuggingFace tutorial.

Source: https://huggingface.co/Ejokhan/HydroGEM
Paper: arXiv 2512.14106
License: CC-BY-NC-4.0  (NON-COMMERCIAL USE ONLY)

Classes here are reproduced verbatim from the public inference tutorial notebook
`HydroGEM_USGS Real Data_With SyntheticAnomalies_Benchmark_Tutorial_matplotlib.ipynb`.
Style was lightly adjusted (no behaviour changes); the published checkpoint
`hydrogem_inference.pt` loads against these definitions exactly.

DO NOT modify the model definitions. Any architecture change here will cause
`load_state_dict` to fail on the published checkpoint.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HydrogemConfig:
    """Model configuration. `from_dict` merges fields from the checkpoint."""

    def __init__(self) -> None:
        self.input_dim = 12
        self.sequence_length = 576
        self.tcn_hidden_dim = 128
        self.tcn_levels = 4
        self.tcn_kernel_size = 3
        self.tcn_dropout = 0.2
        self.transformer_hidden = 256
        self.num_attention_heads = 8
        self.num_transformer_layers = 4
        self.transformer_dropout = 0.1
        self.decoder_hidden = 128
        self.attention_type = "cosine"

    @classmethod
    def from_dict(cls, config_dict: dict) -> HydrogemConfig:
        config = cls()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config


class HydrologicalPositionalEncoding(nn.Module):
    """Positional encoding with hourly + weekly hydrological cycles."""

    def __init__(self, d_model: int, max_len: int = 600, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model - 4, 2).float()
            * -(math.log(10000.0) / (d_model - 4))
        )
        pe[:, 0:-4:2] = torch.sin(position * div_term)
        pe[:, 1:-4:2] = torch.cos(position * div_term)

        hour_in_day = (position % 24) / 24
        day_in_week = (position % 168) / 168
        pe[:, -4] = torch.sin(2 * math.pi * hour_in_day).squeeze()
        pe[:, -3] = torch.cos(2 * math.pi * hour_in_day).squeeze()
        pe[:, -2] = torch.sin(2 * math.pi * day_in_week).squeeze()
        pe[:, -1] = torch.cos(2 * math.pi * day_in_week).squeeze()

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.alpha * self.pe[:, : x.size(1)])


class TemporalBlock(nn.Module):
    """TCN temporal block with residual connection."""

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int,
                 dilation: int, padding: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=1,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.dropout(self.relu(self.bn2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class MultiScaleTCN(nn.Module):
    """Multi-scale Temporal Convolutional Network."""

    def __init__(self, input_size: int, num_channels: list[int],
                 kernel_size: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            padding = (kernel_size - 1) * dilation // 2
            layers.append(TemporalBlock(in_ch, num_channels[i], kernel_size, 1,
                                        dilation, padding, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.permute(0, 2, 1)).permute(0, 2, 1)


class CosineRetentionAttention(nn.Module):
    """Cosine retention attention mechanism."""

    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.ones(heads) * 0.99)
        self.rel_pos_bias = nn.Parameter(torch.zeros(1, heads, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h, d_h = self.heads, self.dim_head
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, n, h, d_h).permute(0, 2, 1, 3)
        k = k.view(b, n, h, d_h).permute(0, 2, 1, 3)
        v = v.view(b, n, h, d_h).permute(0, 2, 1, 3)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        gamma = self.gamma.view(1, h, 1, 1).clamp(0.9, 0.999)
        pos = torch.arange(n, device=x.device)
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
        decay = gamma ** torch.abs(dist).float()
        causal = torch.tril(torch.ones(n, n, device=x.device)).unsqueeze(0).unsqueeze(0)
        sim = sim * decay * causal + self.rel_pos_bias
        sim = sim.masked_fill(
            torch.triu(torch.ones(n, n, device=x.device), 1).bool().unsqueeze(0).unsqueeze(0),
            float("-inf"),
        )
        attn = self.dropout(sim.softmax(dim=-1))
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(b, n, d)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.1,
                 attention_type: str = "cosine") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.use_standard = attention_type != "cosine"
        if self.use_standard:
            self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        else:
            self.attention = CosineRetentionAttention(dim, heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        if self.use_standard:
            attn_out = self.attention(x_norm, x_norm, x_norm)[0]
        else:
            attn_out = self.attention(x_norm)
        x = x + attn_out
        return x + self.ffn(self.norm2(x))


class HydrogemFoundationModel(nn.Module):
    """HydroGEM Foundation Model: TCN encoder -> Transformer -> TCN decoder."""

    def __init__(self, config: HydrogemConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.tcn_hidden_dim)
        self.positional_encoding = HydrologicalPositionalEncoding(
            config.tcn_hidden_dim, config.sequence_length, 0.1
        )
        self.scale_attention = nn.MultiheadAttention(
            config.tcn_hidden_dim, 4, dropout=0.1, batch_first=True,
        )
        tcn_channels = [config.tcn_hidden_dim] * config.tcn_levels
        self.tcn_encoder = MultiScaleTCN(
            config.tcn_hidden_dim, tcn_channels,
            config.tcn_kernel_size, config.tcn_dropout,
        )
        self.tcn_to_transformer = nn.Linear(config.tcn_hidden_dim, config.transformer_hidden)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                config.transformer_hidden, config.num_attention_heads,
                config.transformer_dropout, config.attention_type,
            )
            for _ in range(config.num_transformer_layers)
        ])
        self.transformer_to_tcn = nn.Linear(config.transformer_hidden, config.decoder_hidden)
        self.skip_gate = nn.Parameter(torch.tensor(-2.944))
        decoder_channels = [config.decoder_hidden] * config.tcn_levels
        self.tcn_decoder = MultiScaleTCN(
            config.decoder_hidden, decoder_channels,
            config.tcn_kernel_size, config.tcn_dropout,
        )
        self.reconstruction_head = nn.Linear(config.decoder_hidden, config.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.positional_encoding(self.input_projection(x))
        scale_att, _ = self.scale_attention(proj, proj, proj)
        proj = proj + 0.1 * scale_att
        tcn_feat = self.tcn_encoder(proj)
        trans_in = self.tcn_to_transformer(tcn_feat)
        for layer in self.transformer_layers:
            trans_in = layer(trans_in)
        dec_in = self.transformer_to_tcn(trans_in)
        dec_in = dec_in + torch.sigmoid(self.skip_gate) * tcn_feat
        return self.reconstruction_head(self.tcn_decoder(dec_in))


class MultiScaleAnomalyHead(nn.Module):
    """Multi-scale anomaly detection head."""

    def __init__(self, in_dim: int = 11, hidden_dim: int = 128,
                 scales: tuple[int, ...] = (1, 4), dropout: float = 0.2) -> None:
        super().__init__()
        self.scales = list(scales)
        multi_dim = in_dim * len(scales)
        self.norm = nn.LayerNorm(multi_dim)
        self.mlp = nn.Sequential(
            nn.Linear(multi_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _pool(self, x: torch.Tensor, k: int) -> torch.Tensor:
        if k == 1:
            return x
        B, T, C = x.shape
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, ((k - 1) // 2, (k - 1) - (k - 1) // 2), mode="replicate")
        pooled = F.avg_pool1d(x_pad, k, stride=1)
        return pooled[:, :, :T].transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self._pool(x, k) for k in self.scales], dim=-1)
        return self.mlp(self.norm(x))


# Feature extraction helpers.


def robust_standardize(feat: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    med = torch.nanmedian(feat, dim=1, keepdim=True).values
    mad = torch.nanmedian(torch.abs(feat - med), dim=1, keepdim=True).values
    return torch.clamp((feat - med) / (mad + eps), -10.0, 10.0)


def rolling_std(x: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    B, T, C = x.shape
    x_padded = F.pad(x.transpose(1, 2), (window_size // 2, window_size // 2), mode="reflect")
    return x_padded.unfold(2, window_size, 1).std(dim=-1).transpose(1, 2)


def compute_local_rating_curve_residual(
    log_q: torch.Tensor, log_h: torch.Tensor, window_size: int = 15,
) -> torch.Tensor:
    B, T = log_q.shape
    device, dtype = log_q.device, log_q.dtype
    n = int(window_size)
    if n < 3:
        return log_q.new_zeros(B, T)

    h, q = log_h.unsqueeze(1), log_q.unsqueeze(1)
    ones = torch.ones(1, 1, n, device=device, dtype=dtype)
    padL, padR = (n - 1) // 2, (n - 1) - (n - 1) // 2

    def conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(F.pad(x, (padL, padR), mode="replicate"), ones, stride=1, padding=0)

    sum_h, sum_q = conv(h), conv(q)
    sum_hh, sum_hq = conv(h * h), conv(h * q)
    n_f = float(n)
    mean_h, mean_q = sum_h / n_f, sum_q / n_f
    var_h = (sum_hh - sum_h * mean_h) / n_f
    cov_hq = (sum_hq - sum_h * mean_q) / n_f
    slope = (cov_hq / (var_h + 1e-9)).squeeze(1)
    return torch.clamp_min(1.5 - slope, 0) + torch.clamp_min(slope - 3.0, 0)


def deploy_safe_features(
    out: torch.Tensor, x_obs: torch.Tensor, global_stats: dict,
) -> torch.Tensor:
    q_mu = torch.tensor(global_stats.get("discharge_log_mean", 0.0), device=out.device)
    q_sd = torch.tensor(global_stats.get("discharge_log_std", 1.0), device=out.device)
    h_mu = torch.tensor(global_stats.get("stage_log_mean", 0.0), device=out.device)
    h_sd = torch.tensor(global_stats.get("stage_log_std", 1.0), device=out.device)

    res = torch.abs(out[:, :, 4:6] - x_obs[:, :, 4:6])
    g_out = torch.diff(out[:, :, 4:6], dim=1, prepend=out[:, :1, 4:6])
    g_obs = torch.diff(x_obs[:, :, 4:6], dim=1, prepend=x_obs[:, :1, 4:6])
    roll = rolling_std(res, window_size=7)
    log_q = out[:, :, 4] * q_sd + q_mu
    log_h = out[:, :, 5] * h_sd + h_mu
    rc = compute_local_rating_curve_residual(log_q, log_h, window_size=15).unsqueeze(-1)
    feat = torch.cat([res, g_out, g_obs, roll, rc], dim=-1)
    feat = robust_standardize(feat)
    if feat.size(-1) != 11:
        feat = F.pad(feat, (0, 11 - feat.size(-1)))
    return feat


def morphological_closing(mask: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    mask_float = mask.float().unsqueeze(1)
    dilated = F.max_pool1d(mask_float, kernel_size=kernel_size, stride=1,
                           padding=(kernel_size - 1) // 2)
    eroded = -F.max_pool1d(-dilated, kernel_size=kernel_size, stride=1,
                           padding=(kernel_size - 1) // 2)
    return eroded.squeeze(1).bool()
