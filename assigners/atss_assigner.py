

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from .bbox_utils import iou_similarity as batch_iou_similarity
from .bbox_utils import bbox_center
from .utils import (check_points_inside_bboxes, compute_max_iou_anchor,compute_max_iou_gt)


class ATSSssigner(nn.Module):
    def __int__(self , topk = 9 , num_classes = 80 ,
                force_gt_matching=False,
                eps=1e-9):
        super(ATSSssigner, self).__int__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.eps = eps

    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list,
                             pad_gt_mask):
        pad_gt_mask = pad_gt_mask.tile([1,1,self.topk]).astype(torch.bool)
        gt2anchor_distances_list = torch.split(gt2anchor_distances , num_anchors_list , dim=-1)

        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0,] + num_anchors_index[:-1]

        is_in_topk_list = []
        topk_idxs_list = []
        for distnace , anchors_index in zip(gt2anchor_distances_list , num_anchors_index):
            num_anchors = distnace.shape[-1]
            topk_metrics , topk_idxs = torch.topk(distnace , self.topk, dim=-1,largest=False)

            topk_idxs_list.append(topk_idxs + anchors_index)
            topk_idxs = torch.where(pad_gt_mask , topk_idxs , torch.zeros_like(topk_idxs))

            is_in_topk = F.one_hot(topk_idxs , num_anchors).sum(dim=-2)
            is_in_topk = torch.where(is_in_topk > 1,torch.zeros_like(is_in_topk), is_in_topk)
            is_in_topk_list.append(is_in_topk.to(gt2anchor_distances.dtype))
        is_in_topk_list = torch.concat(is_in_topk_list, dim=-1)
        topk_idxs_list = torch.concat(topk_idxs_list, dim=-1)
        return is_in_topk_list, topk_idxs_list


