import torch
import torch.nn as nn
import torch.nn.functional as F
'''
空间注意力：为每个输入特征图 x, y, z 分别应用空间注意力机制。
逐层融合：先使用 z 更新 y，然后使用更新后的 y 更新 x。
通道注意力：在融合 x 和 y 后，使用通道注意力机制进一步调整特征的重要性。
这种方法结合了空间注意力和通道注意力的优点，有助于更好地融合不同层次的信息。您可以尝试这种融合策略，并根据实际效果调整网络结构和参数。
'''

#无参注意力用作通道注意力
class ChannelSimamModule(nn.Module):
    def __init__(self, channel, e_lambda=1e-4):
        super(ChannelSimamModule, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.channel = channel

    def forward(self, x):
        b, c, h, w = x.size()

        # 计算每个通道的均值
        mu = x.mean(dim=(2, 3), keepdim=True)  # shape: (b, c, 1, 1)

        # 计算每个通道与均值的差的平方
        x_minus_mu_square = (x - mu).pow(2)

        # 计算每个通道的注意力权重
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=1, keepdim=True) / (h * w) + self.e_lambda)) + 0.5

        # 应用注意力权重
        return x * self.act(y)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)



class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=True, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

        self.channel_attention = ChannelSimamModule(in_channels)

    def forward(self, x, y, z):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        # 使用 z 更新 y
        y_q = self.f_y(y)
        x_k = self.f_x(x)

        # 计算基于 y_q 的相似度图
        sim_map = None
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        # 使用 sim_map 更新 y
        y = (1 - sim_map) * y + sim_map * x
 
        # 使用 z 更新 y
        # y = y + z

        # 使用更新后的 y 作为通道权重来更新 x
        x_y_fused = (1 - sim_map) * x + sim_map * y

        # 应用通道注意力
        channel_weight = self.channel_attention(x_y_fused)
        x_y_fused = x_y_fused * channel_weight

        return x_y_fused
