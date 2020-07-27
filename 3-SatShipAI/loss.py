import torch
from segmentation_models_pytorch.utils import functional  as F
from segmentation_models_pytorch.utils.base import Loss, Activation


def bce_weighted(y_pr, y_gt, y_w):
    B = y_pr.shape[0]
    bce2 =  0  # torch.zeros_like(y_pr)
    w = 0
    for i in range(B):
        bce_i = torch.nn.functional.binary_cross_entropy((y_pr[i]).to(torch.float),
                                                         (y_gt[i]).to(torch.float),
                                                         reduction='none')
        w_i = y_w[i]
        bce2 = bce2 + (w_i * bce_i.mean())

        w = w + w_i

    bce2 = bce2 / w

    return bce2


def dice_weighted(y_pr, y_gt, y_w, eps=1., beta=1., activation=None, ignore_channels=None):
    B = y_pr.shape[0]
    act_fun = Activation(activation)
    y_pr = act_fun(y_pr)
    dice = 0
    w = 0
    for i in range(B):
        dice_i = 1 - F.f_score(y_pr[i], y_gt[i], beta=beta, eps=eps, threshold=None, ignore_channels=ignore_channels)
        w_i = y_w[i]
        dice = dice + dice_i * w_i
        w = w + w_i

    dice = dice / w
    return dice


class BCEDiceLossWeighted(Loss):
    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, bce_w=0.6, dice_w=0.4, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.bce_w = bce_w
        self.dice_w = dice_w

    def forward(self, y_pr, y_gt, y_w):
        bce = bce_weighted(y_pr, y_gt, y_w)
        dice = dice_weighted(y_pr, y_gt, y_w, eps=self.eps, beta=self.beta, activation=self.activation,
                             ignore_channels=self.ignore_channels)
        return self.bce_w  * bce + self.dice_w * dice




class BCEDiceLoss(Loss):
    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, bce_w=0.5, dice_w=0.5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.bce_w = bce_w
        self.dice_w = dice_w

    def forward(self, y_pr, y_gt):
        bce = torch.nn.functional.binary_cross_entropy(y_pr, y_gt)
        y_pr = self.activation(y_pr)
        dice = 1 - F.f_score(y_pr, y_gt, beta=self.beta, eps=self.eps, threshold=None,
                             ignore_channels=self.ignore_channels)
        return self.bce_w * bce + self.dice_w * dice

