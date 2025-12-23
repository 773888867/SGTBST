import torch
import torch.nn.functional as F
import torch.nn as nn
from thop.fx_profile import null_print


def half_dict(data):
    data_t1, data_t2 = dict(), dict()
    for k in data.keys():
        data_t1[k] = data[k][::2]
        data_t2[k] = data[k][1::2]
    return data_t1, data_t2

def masking_dict(data, mask_t1, mask_t2):
    data_t1, data_t2 = dict(), dict()
    for k in data.keys():
        data_t1[k] = data[k][mask_t1]
        data_t2[k] = data[k][mask_t2]
    return data_t1, data_t2

def update_dict(src, dst):
    for k in src.keys():
        dst[k] = src[k]

def cxcywh2box(xs, ys, whs):
    return torch.cat([xs - whs[..., 0:1], ys - whs[..., 1:2], xs + whs[..., 2:3], ys + whs[..., 3:4]], dim=-1)

def ind2xy(inds, regs, w=272):
    if inds.dim() == 2:
        inds = inds.unsqueeze(2)
    xs, ys = inds % w, inds // w
    tgt_xys = torch.cat([xs, ys], dim=2) + regs # [B, 500, 2]
    return tgt_xys

def flatten_boxes_dict(boxes):
    if type(boxes).__name__ == 'dict':
        flatten_boxes = {}
        for k in boxes.keys():
            flatten_boxes[k] = boxes[k].flatten(0, 1)
    else:
        flatten_boxes = boxes.flatten(0, 1)
    return flatten_boxes

def mask_boxes_dict(boxes, mask):
    if type(boxes).__name__ == 'dict':
        masked_boxes = {}
        for k in boxes.keys():
            masked_boxes[k] = boxes[k][mask]
    else:
        masked_boxes = boxes[mask]
    return masked_boxes

def xyah2tlbr(xyah):
    xywh = xyah.clone()
    xywh[..., 2] *= xywh[..., 3]
    return torch.cat([ xywh[..., 0] - xywh[..., 2] / 2,
                        xywh[..., 1] - xywh[..., 3] / 2,
                        xywh[..., 0] - xywh[..., 2] / 2,
                        xywh[..., 1] - xywh[..., 3] / 2], dim=-1)

def xywhwh2tlbr(xywhwh):
    return torch.stack([ xywhwh[:, 0] - xywhwh[:, 2],
                         xywhwh[:, 1] - xywhwh[:, 3],
                         xywhwh[:, 0] + xywhwh[:, 4],
                         xywhwh[:, 1] + xywhwh[:, 5]], dim=1)

def tlbr2cxcywh(bboxes):
    ws = bboxes[:, 2] - bboxes[:, 0]
    hs = bboxes[:, 3] - bboxes[:, 1]
    xs = (bboxes[:, 2] + bboxes[:, 0]) / 2
    ys = (bboxes[:, 3] + bboxes[:, 1]) / 2
    return torch.stack([xs, ys, ws, hs], dim=1)

def xyxy2cxcywh(bboxes):
    ws = bboxes[:, 2] - bboxes[:, 0]
    hs = bboxes[:, 3] - bboxes[:, 1]
    xs = (bboxes[:, 2] + bboxes[:, 0]) / 2
    ys = (bboxes[:, 3] + bboxes[:, 1]) / 2
    return torch.stack([xs, ys, ws, hs], dim=1)

def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:  #这里都为true
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]

        # 将特征图fmap的形状调整为(batch, channel, -1)，然后交换维度为(0, 2, 1)，最后确保数据连续存储
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)  # 获取特征图的通道数
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)  # # 根据索引聚集特征
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]  # 使用掩码过滤特征
        fmap = fmap.reshape(-1, dim)   # 重塑特征张量

    return fmap


def split_gather_feature(fmap, index, num_splits=4, mask=None, use_transform=False):
    # print("fmap.size()", fmap.size())
    # 获取特征图的批次大小和通道数
    batch_size, channels, height, width = fmap.shape

    # 首先将特征图调整为 (batch, height * width, channels) 形式
    fmap = fmap.view(batch_size, channels, -1).permute(0, 2, 1)  # 变为 (batch, HxW, channels)
    # print("fmap.size()", 0, fmap.size())

    # 计算每个切分部分的高度
    split_size = height // num_splits  # 假设height能整除num_splits

    # 将特征图沿y轴切分成 num_splits 部分
    split_fmaps = torch.split(fmap, split_size * width, dim=1)  # 每个部分是 (batch, H_part*W, channels)

    # 对每个部分执行gather操作
    gathered_fmaps = []
    for i, split_fmap in enumerate(split_fmaps):
        dim = split_fmap.size(-1)  # 获取通道数 (64)
        # 计算每个分割部分的起始索引偏移量
        offset = i * split_size * width
        # 调整index使其适应当前分割部分的范围
        index_in_split = index - offset
        # 保证索引不越界
        index_in_split = torch.clamp(index_in_split, min=0, max=split_fmap.size(1) - 1)
        index_expanded = index_in_split.unsqueeze(-1).expand(*index_in_split.shape, dim)

        # 根据调整后的索引进行gather操作
        gathered_part = split_fmap.gather(dim=1, index=index_expanded)  # 按索引聚集
        # print("gathered_part.size()", gathered_part.size())

        gathered_fmaps.append(gathered_part)

    # 将每个聚集部分连接起来
    final_fmap = torch.cat(gathered_fmaps, dim=-1)

    return final_fmap

def pseudo_nms(fmap, pool_size=3):
    r"""
    apply max pooling to get the same effect of nms

    Args:
        fmap(Tensor): output tensor of previous step
        pool_size(int): size of max-pooling
    """
    pad = (pool_size - 1) // 2
    fmap_max = F.max_pool2d(fmap, pool_size, stride=1, padding=pad)
    keep = (fmap_max == fmap).float()
    return fmap * keep


def topk_score(scores, K=40, split_num=1):
    """获取分数图中的前K个最高分数点及其相邻点的索引

    Args:
        scores: 形状为(batch, channel, height, width)的4D张量
        K: 需要获取的前K个点的数量,默认为40
        split_num: 返回的索引数量。当split_num=4时,会返回当前点和相邻的3个点(上2下1)

    Returns:
        split_num=1时: 返回5个值(topk_score, topk_inds, topk_clses, base_ys, base_xs)
        split_num!=1时: 返回6个值(topk_score, topk_inds, topk_inds_group, topk_clses, base_ys, base_xs)
        其中topk_inds_group包含所有相邻点的索引数组
    """
    batch, channel, height, width = scores.shape

    # 获取每个特征图上的前K个最高分数及其索引
    topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # 在batch中获取所有通道的前K个最高分数
    topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
    topk_clses = (index / K).int()

    # 获取基础索引
    base_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
    base_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
    base_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

    if split_num == 1:
        # 原始行为：只返回一个索引
        return topk_score, base_inds, topk_clses, base_ys, base_xs
    else:
        # 存储所有索引
        topk_inds_group = []


        # 添加上面相邻的两个点
        curr_ys = torch.clamp(base_ys - 1, 0, height - 1)  # 上面相邻的第一个点
        curr_inds = (curr_ys * width + base_xs).long()
        topk_inds_group.append(curr_inds)

        curr_ys = torch.clamp(base_ys - 2, 0, height - 1)  # 上面相邻的第二个点
        curr_inds = (curr_ys * width + base_xs).long()
        topk_inds_group.append(curr_inds)

        topk_inds_group.append(base_inds)  # 当前点

        # 添加下面相邻的一个点
        curr_ys = torch.clamp(base_ys + 1, 0, height - 1)  # 下面相邻的第一个点
        curr_inds = (curr_ys * width + base_xs).long()
        topk_inds_group.append(curr_inds)

        return topk_score, base_inds, topk_inds_group, topk_clses, base_ys, base_xs