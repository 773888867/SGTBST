import torch
import torch.nn as nn
import torch.nn.functional as F


class PagAFT(nn.Module):
    """
    结合了PagFM的空间注意力机制与AFT_FULL的自注意力机制的模块。
    """

    def __init__(self, d_model, h, w, n=49, simple=False, mid_channels=None, after_relu=False,
                 BatchNorm=nn.BatchNorm2d):
        super(PagAFT, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)

        if simple:
            self.position_biases = torch.zeros((n, n))
        else:
            self.position_biases = nn.Parameter(torch.ones((n, n)))

        self.d_model = d_model
        self.n = n
        self.h = h
        self.w = w
        self.sigmoid = nn.Sigmoid()

        self.f_x = nn.Sequential(
            nn.Conv2d(d_model, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(d_model, mid_channels, kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )

        self.after_relu = after_relu
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        input_size = x.size()

        # 如果需要，在输入之后应用ReLU激活函数
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        # 将输入从 [N, C, H, W] 转换为 [N, H*W, C]
        x = x.view(input_size[0], input_size[1], input_size[2] * input_size[3]).transpose(1, 2)
        y = y.view(input_size[0], input_size[1], input_size[2] * input_size[3]).transpose(1, 2)

        # 计算查询（Query）、键（Key）和值（Value）向量
        q = self.fc_q(x)  # bs, n, dim
        k = self.fc_k(y).view(1, input_size[0], self.n, self.d_model)  # 1, bs, n, dim
        v = self.fc_v(y).view(1, input_size[0], self.n, self.d_model)  # 1, bs, n, dim

        numerator = torch.sum(torch.exp(k + self.position_biases.view(self.n, 1, -1, 1)) * v, dim=2)  # n, bs, dim
        denominator = torch.sum(torch.exp(k + self.position_biases.view(self.n, 1, -1, 1)), dim=2)  # n, bs, dim

        out = (numerator / denominator)  # n, bs, dim
        out = self.sigmoid(q) * (out.permute(1, 0, 2))  # bs, n, dim

        # 将输出从 [N, H*W, C] 转换回 [N, C, H, W]
        out = out.transpose(1, 2).contiguous().view(input_size[0], input_size[1], input_size[2], input_size[3])

        return out


# 示例代码
if __name__ == '__main__':
    block = PagAFT(d_model=64, h=56, w=56, n=56 * 56, mid_channels=32)
    input0 = torch.rand(3, 64, 56, 56)
    input1 = torch.rand(3, 64, 56, 56)
    output = block(input0, input1)
    print(input0.size(), input1.size(), output.size())