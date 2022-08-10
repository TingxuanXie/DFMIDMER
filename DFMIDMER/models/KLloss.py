import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


def kl_loss(feat_real, feat_fit):
    # kl = nn.KLDivLoss(size_average=None)
    real_logit = F.softmax(feat_real, dim=-1)
    _kl_1 = torch.sum(real_logit * (F.log_softmax(feat_real, dim=-1) - F.log_softmax(feat_fit, dim=-1)), 1)
    kl_pos = torch.mean(_kl_1)

    return kl_pos


# x1 = torch.randn(1,8)
# x2 = torch.randn(1,8)
# output = kl_loss(x1, x2)
# print(output)