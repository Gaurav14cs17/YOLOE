import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import ConvBNLayer, BasicBlock


class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, kernel_size=(1, 1))
        self.conv = ConvBNLayer(feat_channels, feat_channels, filter_size=1, )

    def forward(self, feat, avg_feat):
        fc = self.fc(avg_feat)
        weight = F.sigmoid(fc)
        return self.conv(weight * feat)


class PPYOLOEHead(nn.Module):
    def __init__(self, in_channels=None, num_classes=80, ):
        super(PPYOLOEHead, self).__init__()
        if in_channels is None:
            in_channels = [1024, 512, 256]
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes

        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()

        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c))
            self.stem_reg.append(ESEAttn(in_c, ))

        # pred head
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(nn.Conv2d(in_c, self.num_classes, (3, 3), padding=1))
            self.pred_reg.append(nn.Conv2d(in_c, 4 * (self.reg_max + 1), (3, 3), padding=1))

        # projection conv
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, (1, 1), bias=False)

    def forward_train(self, feats):
        # https://arxiv.org/pdf/2203.16250.pdf

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))

            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) + feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat + avg_feat))

            cls_score = F.sigmoid(cls_logit)

            cls_score_list.append(cls_score.flatten(2).transpose([0, 2, 1]))
            reg_distri_list.append(reg_distri.flatten(2).transpose([0, 2, 1]))

        cls_score_list = torch.concat(cls_score_list, dim=1)
        reg_distri_list = torch.concat(reg_distri_list, dim=1)

        return cls_score_list, reg_distri_list

    def forward_eval(self, feats):
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape([-1, 4, self.reg_max + 1, l]).transpose(
                [0, 2, 1, 3])
            reg_dist = self.proj_conv(F.softmax(reg_dist, dim=1))
            # cls and reg
            cls_score = F.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_dist_list.append(reg_dist.reshape([b, 4, l]))

        cls_score_list = torch.concat(cls_score_list, dim=-1)
        reg_dist_list = torch.concat(reg_dist_list, dim=-1)

        return cls_score_list, reg_dist_list

    def forward(self, feats, targets=None):
        if self.training:
            return self.forward_train(feats)
        else:
            return self.forward_eval(feats)
