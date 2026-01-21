import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# utils: patchify + pos-embed
# -----------------------------
def _pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad (right,bottom) so H,W are multiples of `multiple`."""
    _, _, h, w = x.shape
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h))


def _patchify(x: torch.Tensor, patch: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    x: (B,C,H,W) with H,W multiple of patch
    returns tokens: (B, N, C*patch*patch), grid (Ph,Pw)
    """
    b, c, h, w = x.shape
    ph, pw = h // patch, w // patch
    patches = x.unfold(2, patch, patch).unfold(3, patch, patch)  # (B,C,Ph,Pw,p,p)
    patches = patches.contiguous().permute(0, 2, 3, 1, 4, 5)     # (B,Ph,Pw,C,p,p)
    tokens = patches.reshape(b, ph * pw, c * patch * patch)
    return tokens, (ph, pw)


def _interp_abs_pos_embed(
    pos_embed: torch.Tensor,
    base_grid: Tuple[int, int],
    target_grid: Tuple[int, int],
) -> torch.Tensor:
    """
    pos_embed: (1, base_ph*base_pw, D) learned on base_grid
    return: (1, target_ph*target_pw, D)
    """
    base_ph, base_pw = base_grid
    tgt_ph, tgt_pw = target_grid
    d = pos_embed.shape[-1]

    pe = pos_embed.view(1, base_ph, base_pw, d).permute(0, 3, 1, 2)  # (1,D,base_ph,base_pw)
    if (tgt_ph, tgt_pw) != (base_ph, base_pw):
        pe = F.interpolate(pe, size=(tgt_ph, tgt_pw), mode="bilinear", align_corners=False)
    pe = pe.permute(0, 2, 3, 1).contiguous().view(1, tgt_ph * tgt_pw, d)
    return pe

class MotionEncoder(nn.Module):
    """
    Input : (B, T, 2, 96, 96)
    Output: motion tokens (B, Nm, 256)
    """
    def __init__(
        self,
        in_channels=2,
        patch_size=7,
        embed_dim=256,
        num_layers=2,
        nhead=8,
        mlp_ratio=4,
        base_input_hw=(96, 96),
        pool_hw=(9, 9),   # Nm = 81
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.pool_hw = pool_hw

        base_ph = math.ceil(base_input_hw[0] / patch_size)
        base_pw = math.ceil(base_input_hw[1] / patch_size)
        self.base_grid = (base_ph, base_pw)

        self.patch_embed = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, base_ph * base_pw, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * mlp_ratio,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        self.spatial_pool = nn.AdaptiveAvgPool2d(pool_hw)

    def forward(self, mv):
        """
        mv: (BF, T, 2, H, W)   # BF = already concatenated (batch * frames)
        return: (BF, Nm, D)
        """
        BF, T, C, H, W = mv.shape

        # ---- flatten temporal ----
        mv = mv.view(BF * T, C, H, W)   # (BF*T, 2, H, W)

        # ---- preprocessing ----
        mv = _pad_to_multiple(mv, self.patch_size)
        tokens, (ph, pw) = _patchify(mv, self.patch_size)   # (BF*T, P, Cpp)

        x = self.patch_embed(tokens)
        x = x + _interp_abs_pos_embed(
            self.pos_embed, self.base_grid, (ph, pw)
        )

        # ---- encoding ----
        x = self.encoder(x)
        x = self.norm(x)   # (BF*T, P, D)

        # ---- spatial pool (frame-wise, no mixing) ----
        x = x.view(BF * T, ph, pw, self.embed_dim)
        x = x.permute(0, 3, 1, 2)           # (BF*T, D, ph, pw)
        x = self.spatial_pool(x)            # (BF*T, D, oh, ow)

        oh, ow = self.pool_hw
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(BF, T, oh * ow, self.embed_dim)

        # ---- temporal aggregation (per BF) ----
        x = x.mean(dim=1)                   # (BF, Nm, D)

        return x


class MotionProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 1152),
        )

    def forward(self, x):
        return self.proj(x)  # (B, Nm, 1152)


class CrossAttentionFusion(nn.Module):
    """
    Implements:
      F_attn = Attention(Q=F_I, K,V=F_MV)
      F_GOP  = FFN( Bottleneck(F_attn) + F_I )
    """
    def __init__(
        self,
        dim: int = 1152,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        bottleneck_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # residual bottleneck applied to attention output before adding to image
        bn = max(1, dim // bottleneck_ratio)
        self.res_bottleneck = nn.Sequential(
            nn.Linear(dim, bn),
            nn.GELU(),
            nn.Linear(bn, dim),
        )

        # FFN (no residual add after it)
        hidden = dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),           # optional but common; doesn't change "residual count"
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, F_I: torch.Tensor, F_MV: torch.Tensor) -> torch.Tensor:
        """
        F_I : (B, Ni, dim)   image tokens (e.g., (1, 81, 1152))
        F_MV: (B, Nm, dim)   motion tokens projected to same dim (e.g., 1152)
        """
        F_attn, _ = self.attn(
            query=self.norm_q(F_I),
            key=self.norm_kv(F_MV),
            value=self.norm_kv(F_MV),
        )
        
        x = F_I + self.res_bottleneck(F_attn)
        
        return self.ffn(x)

class GOPPipeline(nn.Module):
    def __init__(self):
        super().__init__()

        self.motion_enc = MotionEncoder()
        self.motion_proj = MotionProjector()

    def forward(self, motion_vectors):
        """
        image_tokens  : (B, 81, 1152)  # SigLIP + 2D pool 결과
        motion_vectors: (B, T, 2, 96, 96)
        """

        F_MV = self.motion_enc(motion_vectors)      # (B, Nm, 256)
        F_MV = self.motion_proj(F_MV)               # (B, Nm, 1152)
        
        return F_MV


if __name__ == "__main__":
    B, T = 8, 8
    img_tokens = torch.randn(B, 81, 1152)          # image tokens
    mv = torch.randn(B, T, 2, 96, 96)              # motion vectors

    import time
    start = time.time()
    gop = GOPPipeline()
    out = gop(img_tokens, mv)
    end = time.time()
    print(f"Elapsed time: {end - start:.4f} sec")
    print(out.shape)  # (B, 81, 1152)
