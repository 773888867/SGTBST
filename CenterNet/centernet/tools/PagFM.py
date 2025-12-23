import torch
import torch.nn as nn
import torch.nn.functional as F

class PagFM(nn.Module):
    """
    Position-aware Guided Feature Modulation (PagFM)模块
    该模块通过空间和通道维度上的注意力机制，对输入特征进行加权融合，以增强重要特征并抑制不相关特征。

    参数:
    - in_channels: 输入通道数
    - mid_channels: 中间层通道数
    - after_relu: 是否在输入之后应用ReLU激活函数
    - with_channel: 是否使用通道级别的注意力
    - BatchNorm: 批量归一化层类型，默认为nn.BatchNorm2d
    """
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=True, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        # 1*1卷积，输入([3, 64, 56, 56])，输出([3, 32, 56, 56])
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        # 如果使用通道级别的注意力，则添加一个升维和批量归一化的操作
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )
        # 如果在输入之后应用ReLU激活函数，则添加ReLU操作
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        """
        PagFM模块的前向传播。

        参数:
        - x: 第一个输入特征图
        - y: 第二个输入特征图

        返回:
        - 融合后的特征图
        """
        input_size = x.size()
        # 如果需要，在输入之后应用ReLU激活函数
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        # 计算y的查询向量，并上采样到与x相同的尺寸
        y_q = self.f_y(y)
        # print("y_q1.size", y_q.size())
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=False)
        # print("y_q2.size", y_q.size())
        # 计算x的键向量
        x_k = self.f_x(x)
        # print("x_k.size", x_k.size())
        # print("x_k * y_q.size", (x_k * y_q).size())
        # 根据配置计算空间和/或通道注意力图
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
            # print("sim_map.size",sim_map.size())
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        # 上采样y以匹配x的尺寸，并应用注意力图进行特征融合
        y = F.interpolate(y, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=False)
        x = (1 - sim_map) * x + sim_map * y

        return x

# 示例代码，用于验证PagFM模块的正确性
if __name__ == '__main__':
    block = PagFM(64,32)
    input0 = torch.rand(3, 64, 56, 56)
    input1 = torch.rand(3, 64, 56, 56)
    input2 = torch.rand(3, 64, 56, 56)
    output = block(input1, input2)
    print(input1.size())
    print(input2.size())
    print(output.size())
# ①这里本来是对通道维度进行缩放，我如果不对特征维度，而是对长宽进行缩放呢？
# ②如果in = out，不进行通道转化有影响不
# ③输入三个层，效果会不会更好？这里的net有优势，因为本来就有多个维度的特征图
