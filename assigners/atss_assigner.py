
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from .bbox_utils import iou_similarity as batch_iou_similarity
from .bbox_utils import bbox_center
from .utils import (check_points_inside_bboxes, compute_max_iou_anchor ,compute_max_iou_gt)


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
        pad_gt_mask = pad_gt_mask.tile([1 ,1 ,self.topk]).astype(torch.bool)
        gt2anchor_distances_list = torch.split(gt2anchor_distances , num_anchors_list , dim=-1)

        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0 ,] + num_anchors_index[:-1]

        is_in_topk_list = []
        topk_idxs_list = []
        for distnace , anchors_index in zip(gt2anchor_distances_list , num_anchors_index):
            num_anchors = distnace.shape[-1]
            topk_metrics , topk_idxs = torch.topk(distnace , self.topk, dim=-1 ,largest=False)

            topk_idxs_list.append(topk_idxs + anchors_index)
            topk_idxs = torch.where(pad_gt_mask , topk_idxs , torch.zeros_like(topk_idxs))

            is_in_topk = F.one_hot(topk_idxs , num_anchors).sum(dim=-2)
            is_in_topk = torch.where(is_in_topk > 1 ,torch.zeros_like(is_in_topk), is_in_topk)
            is_in_topk_list.append(is_in_topk.to(gt2anchor_distances.dtype))
        is_in_topk_list = torch.concat(is_in_topk_list, dim=-1)
        topk_idxs_list = torch.concat(topk_idxs_list, dim=-1)
        return is_in_topk_list, topk_idxs_list

    def forward(self ,anchor_bboxes,
                num_anchors_list ,gt_labels,
                gt_bboxes ,pad_gt_mask,
                bg_index ,gt_scores=None ,pred_bboxes=None):

        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py
        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        7. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.

           
        Args:
            anchor_bboxes (Tensor, float32): pre-defined anchors, shape(L, 4),
                    "xmin, xmax, ymin, ymax" format
            num_anchors_list (List): num of anchors in each level
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes,
                    shape(B, n, 1), if None, then it will initialize with one_hot label
            pred_bboxes (Tensor, float32, optional): predicted bounding boxes, shape(B, L, 4)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C), if pred_bboxes is not None, then output ious
        """


        assert gt_labels.ndim == gt_bboxes.ndim and gt_bboxes.ndim == 3

        num_anchors, _ = anchor_bboxes.shape
        batch_size , num_max_boxes ,_ = gt_bboxes.shape
        if num_max_boxes == 0 :
            assigned_labels = torch.full([batch_size , num_anchors], bg_index , dtype=gt_labels.dtype)
            assigned_bboxes = torch.zeros([batch_size,num_anchors, 4])
            assigned_scores = torch.zeros([batch_size , num_anchors, self.num_classes])
            return assigned_labels , assigned_bboxes , assigned_scores



