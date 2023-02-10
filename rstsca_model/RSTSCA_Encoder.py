import torch.nn as nn
from .Sep_STS_Layer import RSTSCABBasicLayer


# ---------------------------------
# Pixel Shuffle
# ---------------------------------
def pixel_shuffle(input, scale_factor):
    if input.dim() == 5:
        B, in_C, D, in_H, in_W = input.shape

        out_C = int(int(in_C / scale_factor) / scale_factor)
        out_H = int(in_H * scale_factor)
        out_W = int(in_W * scale_factor)

        if scale_factor >= 1:
            input_view = input.contiguous().view(B, out_C, D, scale_factor, scale_factor, in_H, in_W)
            shuffle_out = input_view.permute(0, 1, 2, 6, 3, 5, 4).contiguous()
        else:
            block_size = int(1 / scale_factor)
            input_view = input.contiguous().view(B, in_C, D, out_H, block_size, out_W, block_size)
            shuffle_out = input_view.permute(0, 1, 2, 4, 6, 3, 5).contiguous()
        return shuffle_out.view(B, out_C, D, out_H, out_W)
    elif input.dim() == 4:
        B, in_C, in_H, in_W = input.shape

        out_C = int(int(in_C / scale_factor) / scale_factor)
        out_H = int(in_H * scale_factor)
        out_W = int(in_W * scale_factor)

        if scale_factor >= 1:
            input_view = input.contiguous().view(B, out_C, scale_factor, scale_factor, in_H, in_W)
            shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
        else:
            block_size = int(1 / scale_factor)
            input_view = input.contiguous().view(B, in_C, out_H, block_size, out_W, block_size)
            shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return shuffle_out.view(B, out_C, out_H, out_W)

class PixelShuffle(nn.Module):
    """
    Rearrange elements in a tensor of shape (C, rH, rW) to a tensor of shape (r^2*C, H, W)
    """
    def __init__(self, scale_factor, dim, norm_layer=nn.LayerNorm):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor
        self.dim = dim
        self.reduction = nn.Conv3d(2 * dim, dim, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = pixel_shuffle(x, self.scale_factor)
        x = self.reduction(x)

        return x

    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self, dim):
        super().__init__(
            nn.Conv3d(3, dim, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                padding=(1, 1, 1), bias=False),
            nn.ReLU(inplace=False))

class RSTSCABlock(nn.Module):
    def __init__(self, plane, depth, num_frames, num_heads, window_size):
        super(RSTSCABlock, self).__init__()
        self.upper = RSTSCABBasicLayer(plane, depth=depth, num_heads=num_heads,
                                 depth_window_size=window_size, point_window_size=(num_frames, 1, 1))

    def forward(self, x):
        out = self.upper(x)
        return out

class ResBlock(nn.Module):
    def __init__(self, channel, kernel_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(channel, channel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)

        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)

        x += res
        return self.relu(x)

class ResLayer(nn.Module):
    def __init__(self, plane, num_layer, kernel_size=3):
        super(ResLayer, self).__init__()
        self.layer = nn.ModuleList()
        for i in range(num_layer):
            self.layer.append(ResBlock(plane, kernel_size=kernel_size))

        self.layer = nn.Sequential(*self.layer)

    def forward(self, x):
        return self.layer(x)


class RSTSCAGroup_Encoder(nn.Module):

    def __init__(self, nf, NF, window_size, nh):
        super(RSTSCAGroup_Encoder, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=nf[-1]//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            ResBlock(nf[-1]//2, kernel_size=3),
        )

        self.stage1 = RSTSCABlock(nf[-1], depth=2, num_frames=NF, num_heads=nh[0], window_size=window_size[0])
        self.stage2 = RSTSCABlock(nf[-2], depth=2, num_frames=NF, num_heads=nh[1], window_size=window_size[1])
        self.stage3 = RSTSCABlock(nf[-3], depth=6, num_frames=NF, num_heads=nh[2], window_size=window_size[2])
        self.stage4 = RSTSCABlock(nf[-4], depth=2, num_frames=NF, num_heads=nh[3], window_size=window_size[3])

        self.down0 = PixelShuffle(scale_factor=1 / 2, dim=nf[-1])
        self.down1 = PixelShuffle(scale_factor=1 / 2, dim=nf[-2])
        self.down2 = PixelShuffle(scale_factor=1 / 2, dim=nf[-3])
        self.down3 = PixelShuffle(scale_factor=1 / 2, dim=nf[-4])

    def forward(self, x):
        x0 = self.stem(x)

        x1 = self.down0(x0)
        x1 = self.stage1(x1)

        x2 = self.down1(x1)
        x2 = self.stage2(x2)

        x3 = self.down2(x2)
        x3 = self.stage3(x3)

        x4 = self.down3(x3)
        x4 = self.stage4(x4)

        return x0, x1, x2, x3, x4

