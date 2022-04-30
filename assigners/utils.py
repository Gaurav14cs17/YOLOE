import torch
import torch.nn.functional as F
import torch.nn as nn


def generate_anchors_for_grid_cell(feats,fpn_strides,grid_cell_size=5.0,grid_cell_offset=0.5):
    anchors = []
    anchors_points = []
    num_anchors_list = []
    stride_tensor = []
    for feat, stride in zip(feats, fpn_strides):
        _, _, h, w = feat.shape
        cell_half_size = (grid_cell_size * stride) * 0.5

        shift_x = (torch.arange(end=w) + grid_cell_offset) * stride
        shift_y = (torch.arange(end=h) + grid_cell_offset) * stride

        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        anchor = torch.stack([shift_x - cell_half_size, shift_y - cell_half_size,
                              shift_x + cell_half_size, shift_y + cell_half_size], dim=-1).type(feat.dtype)

        anchor_point = torch.stack([shift_x, shift_y], dim=-1).type(feat.dtype)

        anchors.append(anchor.reshape([-1, 4]))
        anchors_points.append(anchor_point.reshape([-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=feat.dtype)))

    anchors = torch.concat(anchors)
    anchors.requires_grad = False

    anchors_points = torch.concat(anchors_points)
    anchors_points.requires_grad = False

    stride_tensor = torch.concat(stride_tensor)
    stride_tensor.requires_grad = False

    return anchors, anchors_points, num_anchors_list, stride_tensor
