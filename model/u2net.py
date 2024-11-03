import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from model.fusion_mamba import FusionMamba
from mamba_ssm.modules.mamba_simple import Mamba


class ResBlock2D(nn.Module):
    def __init__(self, dim, res_se_ratio):
        super().__init__()
        hidden_dim = int(res_se_ratio * dim)
        self.conv0 = nn.Conv2d(dim, hidden_dim, 3, 1, 1)
        self.conv1 = nn.Conv2d(hidden_dim, dim, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv0(x))
        rs1 = self.conv1(rs1)
        rs = torch.add(x, rs1)
        return rs


class PixelShuffle(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.upsamle = nn.Sequential(
            nn.Conv2d(dim, dim*(scale**2), 3, 1, 1, bias=False),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        return self.upsamle(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale, upsample='default', combine='add'):
        super().__init__()
        if upsample == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        elif upsample == 'bicubic':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        elif upsample == 'pixelshuffle':
            self.up = nn.Sequential(
                PixelShuffle(in_channels, scale),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, scale, scale, 0),
                nn.LeakyReLU()
            )
        if combine == 'concat':
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels * 2, 3, 1, 1, groups=out_channels*2),
                nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0),
                nn.LeakyReLU()
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.Conv2d(out_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
                )
        self.combine = combine

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.combine == 'concat':
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1 + x2
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, scale, downsample='default'):
        super().__init__()
        if downsample == 'maxpooling':
            self.down = nn.Sequential(
                nn.MaxPool2d(scale),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.LeakyReLU()
            )
        else:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, scale, scale, 0),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.down(x)


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, H, W, scale=2, sample_mode='down'):
        super().__init__()
        self.fm = FusionMamba(in_channels, H, W)
        if sample_mode == 'down':
            self.sample = Down(in_channels, out_channels, scale)
        elif sample_mode == 'up':
            self.sample = Up(in_channels, out_channels, scale)

    def forward(self, pan, ms, pan_pre=None, ms_pre=None):
        pan, ms = self.fm(pan, ms)
        if pan_pre is None:
            pan_skip = pan
            ms_skip = ms
            pan = self.sample(pan)
            ms = self.sample(ms)
            return pan, ms, pan_skip, ms_skip
        else:
            pan = self.sample(pan, pan_pre)
            ms = self.sample(ms, ms_pre)
            return pan, ms


class SpeAttention(nn.Module):
    def __init__(self, spe_channels, se_ratio=8, mode='mamba', channels=32):
        super().__init__()
        
        self.mode = mode
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        if mode == 'mamba':
            self.block = nn.Sequential(
                nn.Linear(1, channels),
                nn.LayerNorm(channels),
                Mamba(channels, expand=1, d_state=8, bimamba_type='v2', if_devide_out=True, use_norm=True),
                nn.Linear(channels, 1)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(spe_channels, spe_channels // se_ratio, 1, 1, 0, bias=False),
                nn.LeakyReLU(),
                nn.Conv2d(spe_channels // se_ratio, spe_channels, 1, 1, 0, bias=False),
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = self.pooling(input)
        if self.mode == 'mamba':
            output = self.block(input.squeeze(-1)).unsqueeze(-1)
        else:
            output = self.block(input)
        return self.sigmoid(output)


class U2Net(nn.Module):
    def __init__(self, dim, pan_dim, ms_dim, H=64, W=64, scale=4):
        super().__init__()

        self.upsample = PixelShuffle(ms_dim, scale)
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(pan_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(ms_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.to_hrms = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, ms_dim, 3, 1, 1)
        )

        # dimension for each stage
        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)

        # main body
        self.stage0 = Stage(dim0, dim1, H, W, sample_mode='down')
        self.stage1 = Stage(dim1, dim2, H//2, W//2, sample_mode='down')
        self.stage2 = Stage(dim2, dim1, H//4, W//4, sample_mode='up')
        self.stage3 = Stage(dim1, dim0, H//2, W//2, sample_mode='up')
        self.stage4 = FusionMamba(dim0, H, W, final=True)

        self.spe_attn = SpeAttention(ms_dim, 16, 'mamba', dim)

    def forward(self, ms, pan):
        lrms = ms
        ms = self.upsample(ms)
        skip = ms
        pan = self.raise_pan_dim(pan)
        ms = self.raise_ms_dim(ms)

        # main body
        pan, ms, pan_skip0, ms_skip0 = self.stage0(pan, ms)
        pan, ms, pan_skip1, ms_skip1 = self.stage1(pan, ms)
        pan, ms = self.stage2(pan, ms, pan_skip1, ms_skip1)
        pan, ms = self.stage3(pan, ms, pan_skip0, ms_skip0)
        output = self.stage4(pan, ms)

        spe_attn = self.spe_attn(lrms)

        output = self.to_hrms(output) * spe_attn + skip
        return output
