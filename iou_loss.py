from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import math
import torch

__all__ = ['IouLoss', 'GIoULoss', 'DIouLoss']


def iou_similarity(box1, box2, eps=1e-9):
    """Calculate iou of box1 and box2
    Args:
        box1 (Tensor): box with the shape [N, M1, 4]
        box2 (Tensor): box with the shape [N, M2, 4]
    Return:
        iou (Tensor): iou between box1 and box2 with the shape [N, M1, M2]
    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps
    return overlap / union


def bbox_iou(box1, box2, giou=False, diou=False, ciou=False, eps=1e-9):
    """calculate the iou of box1 and box2
    Args:
        box1 (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        box2 (list): [x, y, w, h], all have the shape [b, na, h, w, 1]
        giou (bool): whether use giou or not, default False
        diou (bool): whether use diou or not, default False
        ciou (bool): whether use ciou or not, default False
        eps (float): epsilon to avoid divide by zero
    Return:
        iou (Tensor): iou of box1 and box1, with the shape [b, na, h, w, 1]
    """
    px1, py1, px2, py2 = box1
    gx1, gy1, gx2, gy2 = box2
    x1 = torch.maximum(px1, gx1)
    y1 = torch.maximum(py1, gy1)
    x2 = torch.minimum(px2, gx2)
    y2 = torch.minimum(py2, gy2)
    overlap = ((x2 - x1).clip(0)) * ((y2 - y1).clip(0))
    area1 = (px2 - px1) * (py2 - py1)
    area1 = area1.clip(0)
    area2 = (gx2 - gx1) * (gy2 - gy1)
    area2 = area2.clip(0)
    union = area1 + area2 - overlap + eps
    iou = overlap / union
    if giou or ciou or diou:
        # convex w, h
        cw = torch.maximum(px2, gx2) - torch.minimum(px1, gx1)
        ch = torch.maximum(py2, gy2) - torch.minimum(py1, gy1)
        if giou:
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
        else:
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + eps
            # center distance
            rho2 = ((px1 + px2 - gx1 - gx2) ** 2 + (py1 + py2 - gy1 - gy2) ** 2) / 4
            if diou:
                return iou - rho2 / c2
            else:
                w1, h1 = px2 - px1, py2 - py1 + eps
                w2, h2 = gx2 - gx1, gy2 - gy1 + eps
                delta = torch.atan(w1 / h1) - torch.atan(w2 / h2)
                v = (4 / math.pi ** 2) * torch.pow(delta, 2)
                alpha = v / (1 + eps - iou + v)
                alpha.stop_gradient = True
                return iou - (rho2 / c2 + v * alpha)
    else:
        return iou


class IouLoss(object):
    """
    iou loss, see https://arxiv.org/abs/1908.03851
    loss = 1.0 - iou * iou
    Args:
        loss_weight (float): iou loss weight, default is 2.5
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
        ciou_term (bool): whether to add ciou_term
        loss_square (bool): whether to square the iou term
    """
    def __init__(self, loss_weight=2.5, giou=False, diou=False, ciou=False, loss_square=True):
        self.loss_weight = loss_weight
        self.giou = giou
        self.diou = diou
        self.ciou = ciou
        self.loss_square = loss_square

    def __call__(self, pbox, gbox):
        iou = bbox_iou(pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
        if self.loss_square:
            loss_iou = 1 - iou * iou
        else:
            loss_iou = 1 - iou
        loss_iou = loss_iou * self.loss_weight
        return loss_iou


class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1., eps=1e-10, reduction='none'):
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def bbox_overlap(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        xkis1 = torch.maximum(x1, x1g)
        ykis1 = torch.maximum(y1, y1g)
        xkis2 = torch.minimum(x2, x2g)
        ykis2 = torch.minimum(y2, y2g)
        w_inter = (xkis2 - xkis1).clip(0)
        h_inter = (ykis2 - ykis1).clip(0)
        overlap = w_inter * h_inter
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union
        return iou, overlap, union

    def __call__(self, pbox, gbox, iou_weight=1., loc_reweight=None):
        x1, y1, x2, y2 = torch.split(pbox, split_size_or_sections = 4, dim=-1)
        x1g, y1g, x2g, y2g = torch.split(gbox, split_size_or_sections =4, dim=-1)
        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
        xc1 = torch.minimum(x1, x1g)
        yc1 = torch.minimum(y1, y1g)
        xc2 = torch.maximum(x2, x2g)
        yc2 = torch.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = torch.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh) * miou - loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == 'none':
            loss = giou
        elif self.reduction == 'sum':
            loss = torch.sum(giou * iou_weight)
        else:
            loss = torch.mean(giou * iou_weight)
        return loss * self.loss_weight


class DIouLoss(GIoULoss):
    """
    Distance-IoU Loss, see https://arxiv.org/abs/1911.08287
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        use_complete_iou_loss (bool): whether to use complete iou loss
    """

    def __init__(self, loss_weight=1., eps=1e-10, use_complete_iou_loss=True):
        super(DIouLoss, self).__init__(loss_weight=loss_weight, eps=eps)
        self.use_complete_iou_loss = use_complete_iou_loss

    def __call__(self, pbox, gbox, iou_weight=1. , loc_reweight = None ):
        x1, y1, x2, y2 = torch.split(pbox, split_size_or_sections =4, dim=-1)
        x1g, y1g, x2g, y2g = torch.split(gbox, split_size_or_sections =4, dim=-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        cxg = (x1g + x2g) / 2
        cyg = (y1g + y2g) / 2
        wg = x2g - x1g
        hg = y2g - y1g

        x2 = torch.maximum(x1, x2)
        y2 = torch.maximum(y1, y2)

        # A and B
        xkis1 = torch.maximum(x1, x1g)
        ykis1 = torch.maximum(y1, y1g)
        xkis2 = torch.minimum(x2, x2g)
        ykis2 = torch.minimum(y2, y2g)

        # A or B
        xc1 = torch.minimum(x1, x1g)
        yc1 = torch.minimum(y1, y1g)
        xc2 = torch.maximum(x2, x2g)
        yc2 = torch.maximum(y2, y2g)

        intsctk = (xkis2 - xkis1) * (ykis2 - ykis1)
        intsctk = intsctk * torch.gt(xkis2, xkis1) * torch.gt(ykis2, ykis1)
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsctk + self.eps
        iouk = intsctk / unionk

        # DIOU term
        dist_intersection = (cx - cxg) * (cx - cxg) + (cy - cyg) * (cy - cyg)
        dist_union = (xc2 - xc1) * (xc2 - xc1) + (yc2 - yc1) * (yc2 - yc1)
        diou_term = (dist_intersection + self.eps) / (dist_union + self.eps)

        # CIOU term
        ciou_term = 0
        if self.use_complete_iou_loss:
            ar_gt = wg / hg
            ar_pred = w / h
            arctan = torch.atan(ar_gt) - torch.atan(ar_pred)
            ar_loss = 4. / np.pi / np.pi * arctan * arctan
            alpha = ar_loss / (1 - iouk + ar_loss + self.eps)
            alpha.stop_gradient = True
            ciou_term = alpha * ar_loss

        diou = torch.mean((1 - iouk + ciou_term + diou_term) * iou_weight)
        return diou * self.loss_weight
