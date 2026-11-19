"""GeoLitterNet v3 architecture.

Copied verbatim from the training notebook "GeoLitterNet New V2" (Kaggle) so that the released
checkpoint (geolitternet_v3_fair_seed_42_best.pth) loads with strict=True. Do not change layer names
or shapes here: any edit breaks weight loading. Behavioural changes belong in preprocess.py / engine.py.

Input : (B, 16, H, W) = 11 Sentinel-2 bands (B1,B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12) + 5 physics indices
Output: (B, 15, H, W) logits over the MARIDA classes (index 0 = Marine Debris). H, W must be divisible by 16.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)

class HighResPhysicsTokenizer(nn.Module):
    def __init__(self, in_raw_channels=11, in_idx_channels=5, out_channels=32):
        super().__init__()
        self.raw_stem = nn.Sequential(
            nn.Conv2d(in_raw_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.idx_stem = nn.Sequential(
            nn.Conv2d(in_idx_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=out_channels),
            nn.GELU()
        )
        self.gate_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.proj_stem = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x_16):
        x_raw = x_16[:, :11, :, :]
        x_idx = x_16[:, 11:, :, :]

        f_raw = self.raw_stem(x_raw)
        f_idx = self.idx_stem(x_idx)

        g_phys = torch.sigmoid(self.gate_conv(f_idx))
        f_gated = f_raw * g_phys

        s_0 = self.proj_stem(torch.cat([f_raw, f_gated], dim=1))  # 256x256 Full-Resolution Skip
        z_0 = self.downsample(s_0)                                  # 128x128 Token Grid
        return z_0, s_0, x_idx

class DynamicLocalGlobal_SR_MHSA(nn.Module):
    def __init__(self, dim, num_heads=4, sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.local_path = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwconv_ffn = nn.Conv2d(dim * 4, dim * 4, kernel_size=3, padding=1, groups=dim * 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
        f_local = self.local_path(x_2d).flatten(2).transpose(1, 2)

        x_norm = self.norm1(x)
        q = self.q(x_norm).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_spatial = x_norm.permute(0, 2, 1).reshape(B, C, H, W)
            x_sr = self.sr(x_spatial).reshape(B, C, -1).permute(0, 2, 1)
            x_sr = self.norm(x_sr)
            kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x_norm).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        f_global = (attn @ v).transpose(1, 2).reshape(B, N, C)

        gate = torch.sigmoid(self.alpha)
        x = x + (gate * f_local + (1.0 - gate) * f_global)

        x_norm2 = self.norm2(x)
        mlp_in = self.mlp[0](x_norm2)
        mlp_spatial = mlp_in.permute(0, 2, 1).reshape(B, C * 4, H, W)
        mlp_spatial = self.dwconv_ffn(mlp_spatial)
        mlp_in = mlp_spatial.reshape(B, C * 4, -1).permute(0, 2, 1)
        x_ffn = self.mlp[2](self.mlp[1](mlp_in))

        return x + x_ffn

class MS_PGSS_Gate(nn.Module):
    def __init__(self, in_channels, physics_channels=5):
        super().__init__()
        self.phys_proj = nn.Sequential(
            nn.Conv2d(physics_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=in_channels),
            nn.GELU()
        )
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.GroupNorm(num_groups=4, num_channels=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, e_i, x_idx):
        H_i, W_i = e_i.shape[2], e_i.shape[3]
        p_i = F.interpolate(self.phys_proj(x_idx), size=(H_i, W_i), mode='bilinear', align_corners=False)
        g_i = self.gate_conv(torch.cat([e_i, p_i], dim=1))
        e_fused = e_i + self.gamma * (p_i * g_i)
        return e_fused

class MultiScaleConvDecoder(nn.Module):
    def __init__(self, encoder_dims=[32, 64, 128, 256], num_classes=15):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(encoder_dims[3], encoder_dims[2], kernel_size=2, stride=2)
        self.conv1 = DoubleConvBlock(encoder_dims[2] * 2, encoder_dims[2])

        self.up2 = nn.ConvTranspose2d(encoder_dims[2], encoder_dims[1], kernel_size=2, stride=2)
        self.conv2 = DoubleConvBlock(encoder_dims[1] * 2, encoder_dims[1])

        self.up3 = nn.ConvTranspose2d(encoder_dims[1], encoder_dims[0], kernel_size=2, stride=2)
        self.conv3 = DoubleConvBlock(encoder_dims[0] * 2, encoder_dims[0])

        self.up4 = nn.ConvTranspose2d(encoder_dims[0], 32, kernel_size=2, stride=2)
        self.conv4 = DoubleConvBlock(32 + 32, 32)

        self.cls_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, e1, e2, e3, e4, s0):
        d1 = self.conv1(torch.cat([self.up1(e4), e3], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d1), e2], dim=1))
        d3 = self.conv3(torch.cat([self.up3(d2), e1], dim=1))
        d4 = self.conv4(torch.cat([self.up4(d3), s0], dim=1))
        return self.cls_head(d4)

class GeoLitterNet(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.pst = HighResPhysicsTokenizer(in_raw_channels=11, in_idx_channels=5, out_channels=32)

        self.stage1 = DynamicLocalGlobal_SR_MHSA(dim=32, num_heads=2, sr_ratio=8)
        self.down2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.stage2 = DynamicLocalGlobal_SR_MHSA(dim=64, num_heads=4, sr_ratio=4)
        self.gate2 = MS_PGSS_Gate(in_channels=64)

        self.down3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.stage3 = DynamicLocalGlobal_SR_MHSA(dim=128, num_heads=8, sr_ratio=2)
        self.gate3 = MS_PGSS_Gate(in_channels=128)

        self.down4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.stage4 = DynamicLocalGlobal_SR_MHSA(dim=256, num_heads=8, sr_ratio=1)

        self.decoder = MultiScaleConvDecoder(encoder_dims=[32, 64, 128, 256], num_classes=num_classes)

    def forward(self, x_16):
        z0, s0, x_idx = self.pst(x_16)
        B, C1, H1, W1 = z0.shape

        e1 = self.stage1(z0.flatten(2).transpose(1, 2), H1, W1).transpose(1, 2).reshape(B, C1, H1, W1)

        e2_in = self.down2(e1)
        H2, W2 = e2_in.shape[2], e2_in.shape[3]
        e2 = self.stage2(e2_in.flatten(2).transpose(1, 2), H2, W2).transpose(1, 2).reshape(B, 64, H2, W2)
        e2 = self.gate2(e2, x_idx)

        e3_in = self.down3(e2)
        H3, W3 = e3_in.shape[2], e3_in.shape[3]
        e3 = self.stage3(e3_in.flatten(2).transpose(1, 2), H3, W3).transpose(1, 2).reshape(B, 128, H3, W3)
        e3 = self.gate3(e3, x_idx)

        e4_in = self.down4(e3)
        H4, W4 = e4_in.shape[2], e4_in.shape[3]
        e4 = self.stage4(e4_in.flatten(2).transpose(1, 2), H4, W4).transpose(1, 2).reshape(B, 256, H4, W4)

        return self.decoder(e1, e2, e3, e4, s0)

