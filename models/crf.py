import nptyping
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax, unary_from_labels
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import to_3tuple


class CRF(nn.Module):
    def __init__(self, num_iters, num_classes, bilateral_weight=1, gaussian_weight=2,
                 gaussian_spatial_sigma=3, bilateral_spatial_sigma=3, bilateral_intensity_sigma=10):
        super().__init__()
        self.num_iters = num_iters
        self.num_classes = num_classes
        self.bilateral_wight = bilateral_weight
        self.gaussian_weight = gaussian_weight
        self.gaussian_sigma = to_3tuple(gaussian_spatial_sigma)
        self.bilateral_spatial_sigma = to_3tuple(bilateral_spatial_sigma)
        self.bilateral_intensity_sigma = (bilateral_intensity_sigma,)

    def forward(self, logits, reference):
        b, c, depth, height, width = logits.shape
        # depth, height, width = logits.shape
        assert b == 1 and c == self.num_classes
        logits = logits.squeeze(0)  # c, d, h, w
        reference = reference.squeeze(0).squeeze(0)  # d, h, w

        Q = logits
        # Q = torch.argmax(logits, dim=0).cpu()
        # print(Q.shape, Q.dtype, Q.min(), Q.max())

        U = unary_from_softmax(Q)
        # U = unary_from_labels(logits, 2, gt_prob=0.9, zero_unsure=False)
        # print(U.shape, U.min(), U.max())
        gaussian_pairwise = create_pairwise_gaussian(sdims=self.gaussian_sigma, shape=reference.shape)
        bilateral_pairwise = create_pairwise_bilateral(sdims=self.bilateral_spatial_sigma, schan=self.bilateral_intensity_sigma, img=reference)

        d = dcrf.DenseCRF(depth * height * width, self.num_classes)
        d.setUnaryEnergy(U)
        d.addPairwiseEnergy(gaussian_pairwise, compat=self.gaussian_weight, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        d.addPairwiseEnergy(bilateral_pairwise, compat=self.bilateral_wight, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q, tmp1, tmp2 = d.startInference()
        min_kl = np.inf
        _iter = 0
        while True:
            d.stepInference(Q, tmp1, tmp2)
            kl = d.klDivergence(Q) / (depth * height * width)
            if kl < min_kl and self.num_iters < _iter:
                min_kl = kl
                _iter += 1
            else:
                break

        out = np.array(Q)
        out = torch.FloatTensor(out).reshape(-1, depth, height, width)

        return out.unsqueeze(0)
