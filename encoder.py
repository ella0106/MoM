from .utils.utils import *

# ----------------------------
# Frame encoder
# ----------------------------
class FrameEncoder(nn.Module):
    def __init__(self, in_chans, hidden_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, hidden_dim)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feat = self.cnn(x).flatten(1)  # (B*T, 64)
        feat = self.fc(feat)           # (B*T, D)
        return feat.view(B, T, -1)     # (B, T, D)

# ----------------------------
# Positional encoding (GOP + Frame index)
# ----------------------------
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, num_gops, num_frames, dim):
        super().__init__()
        self.gop_embed = nn.Embedding(num_gops, dim)
        self.frame_embed = nn.Embedding(num_frames, dim)

    def forward(self, gop_ids, frame_ids):
        gop_pos = self.gop_embed(gop_ids).unsqueeze(1)     # (B, 1, D)
        frame_pos = self.frame_embed(frame_ids)            # (B, T, D)
        return gop_pos, frame_pos

# ----------------------------
# Transformer blocks
# ----------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, depth=2, heads=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.transformer(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, q, kv):
        out, _ = self.attn(q, kv, kv)
        return out

# ----------------------------
# 전체 모델
# ----------------------------
class MVResidualModel(nn.Module):
    def __init__(self, in_chans_mv=2, in_chans_res=3, dim=768, num_gops=500, num_frames=12, depth=2, heads=8):
        super().__init__()
        
        self.mv_encoder = FrameEncoder(in_chans_mv, dim)
        self.res_encoder = FrameEncoder(in_chans_res, dim)

        self.pos_enc = TemporalPositionalEncoding(num_gops, num_frames, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))  # GOP CLS

        self.mv_self = SelfAttentionBlock(dim, depth, heads)
        self.res_self = SelfAttentionBlock(dim, depth, heads)

        self.cross_mv2res = CrossAttentionBlock(dim, heads)
        self.cross_res2mv = CrossAttentionBlock(dim, heads)
        
        self.fc = nn.Linear(dim, 1152)

    def forward(self, mv_frames, res_frames):
        """
        mv_frames:  (B, T, 2, H, W)
        res_frames: (B, T, 3, H, W)
        gop_ids:    (B,) GOP index
        """
        B, T, _, _, _ = mv_frames.shape
        gop_ids = torch.arange(B, device=mv_frames.device)  # (B,)
        frame_ids = torch.arange(T, device=mv_frames.device).unsqueeze(0).repeat(B, 1)  # (B,T)

        # 1. Frame encoding
        mv_feat = self.mv_encoder(mv_frames)    # (B, T, D)
        res_feat = self.res_encoder(res_frames) # (B, T, D)

        # 2. Add GOP CLS + positional embedding
        gop_pos, frame_pos = self.pos_enc(gop_ids, frame_ids)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        mv_feat = torch.cat([cls_tokens, mv_feat], dim=1) + torch.cat([gop_pos, frame_pos], dim=1)
        res_feat = torch.cat([cls_tokens, res_feat], dim=1) + torch.cat([gop_pos, frame_pos], dim=1)

        # 3. Self-attention per branch
        mv_feat = self.mv_self(mv_feat)    # (B, T+1, D)
        res_feat = self.res_self(res_feat) # (B, T+1, D)

        # 4. Cross-attention
        mv2res = self.cross_mv2res(res_feat, mv_feat)  # (B, T+1, D)
        # res2mv = self.cross_res2mv(mv_feat, res_feat)  # (B, T+1, D)
        mv2res = self.fc(mv2res) # (B, T+1, 1152)

        # 5. 최종 feature
        # fused = torch.cat([mv2res, res2mv], dim=-1)    # (B, T+1, 2D)
        fused = mv2res
        
        return fused
    
class EarlyFusionProjector(nn.Module):
    def __init__(self, in_chans=8, out_chans=3):
        super().__init__()
        # naive: just 1x1 conv to collapse channels
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=1)

    def forward(self, rgb, mv, res):
        """
        rgb: (T,H,W,3)
        mv:  (T,H,W,2)
        res: (T,H,W,3)
        """
        fused = np.concatenate([rgb, mv, res], axis=-1)  # (T,H,W,8)
        fused = torch.from_numpy(fused).permute(0,3,1,2).float()  # (T,8,H,W)
        fused = self.proj(fused)  # (T,3,H,W)
        return fused

class LateFusionProjector(nn.Module):
    def __init__(self, dim_rgb, dim_mvres, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_rgb + dim_mvres, dim_out)

    def forward(self, rgb_feat, mvres_feat):
        """
        rgb_feat:   (B, T, D_rgb)  # frame encoder output (LLaVA vision tower output or pooled feature)
        mvres_feat: (B, T, D_mvres) # MVResidualModel의 frame token (CLS 제외)
        """
        assert rgb_feat.size(1) == mvres_feat.size(1), "Frame alignment mismatch!"

        fused = torch.cat([rgb_feat, mvres_feat], dim=-1)  # (B, T, D_rgb+D_mvres)
        fused = self.proj(fused)                           # (B, T, D_out)
        return fused

# ----------------------------
# 위에서 정의한 MVResidualModel 불러오기
# ----------------------------
# (코드를 한 파일에 넣는다면 여기 import 대신 class 정의 붙여넣기)

if __name__ == "__main__":
    # 가짜 입력 생성
    B, T, H, W = 2, 12, 224, 224   # batch=2, GOP=12프레임
    mv_frames = torch.randn(B, T, 2, H, W)    # motion vector (dx, dy)
    res_frames = torch.randn(B, T, 3, H, W)   # residual (3채널)
    gop_ids = torch.arange(B)            # 배치별 GOP index

    # 모델 초기화
    model = MVResidualModel(
        in_chans_mv=2,
        in_chans_res=3,
        dim=256,         # 테스트라서 작은 dim
        num_gops=B,
        num_frames=12,
        depth=2,
        heads=4
    )

    # forward
    out = model(mv_frames, res_frames, gop_ids)

    print("Output shape:", out.shape)   # (B, T+1, 2D)
    print("첫 번째 CLS 토큰 shape:", out[:, 0].shape)  # (B, 2D)
    print("나머지 frame 토큰 shape:", out[:, 1:].shape) # (B, T, 2D)
