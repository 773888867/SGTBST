# ---------------------------------------
# 论文: “Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks.” ArXiv abs/1905.09646
# Github地址: https://github.com/implus/PytorchInsight
# ---------------------------------------
import torch
from torch import nn


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups = 64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = nn.Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x ,y): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)

        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        xt = xn.view(b * self.groups, -1)
        xt = xt - xt.mean(dim=1, keepdim=True)
        std = xt.std(dim=1, keepdim=True) + 1e-5
        xt = xt / std
        xt = xt.view(b, self.groups, h, w)
        xt = xt * self.weight + self.bias
        xt = xt.view(b * self.groups, 1, h, w)
        # print("t.size",xt.size())

        y = y.view(b * self.groups, -1, h, w)
        yn = x * self.avg_pool(x)
        yn = yn.sum(dim=1, keepdim=True)
        yt = yn.view(b * self.groups, -1)
        yt = yt - yt.mean(dim=1, keepdim=True)
        std = yt.std(dim=1, keepdim=True) + 1e-5
        yt = yt / std
        yt = yt.view(b, self.groups, h, w)
        yt = yt * self.weight + self.bias
        yt = yt.view(b * self.groups, 1, h, w)
        # print("t.size", yt.size())

        sim_map = torch.sigmoid(xt * yt)
        x = (1 - sim_map) * x + sim_map * y

        x = x.view(b, c, h, w)
        return x


#   输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.randn(3, 32, 64, 64)
    input2 = torch.randn(3, 32, 64, 64)
    sge = SpatialGroupEnhance(groups=4)
    output = sge(input,input2)
    print(output.shape)
