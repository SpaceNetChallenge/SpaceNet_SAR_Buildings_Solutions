import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.nn import BCEWithLogitsLoss


def get_loss(loss_name='cce'):
    if loss_name == 'focal_dice':
        return FocalDiceLoss()
    elif loss_name == 'bce':
        return BCEWithLogitsLoss()
    

class FocalDiceLoss(torch.nn.Module):
    def __init__(self, coef_focal=1.0, coef_dice=1.0, weights=(1.0, 0.1, 0.5)):
        super().__init__()
       
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.weights = weights

        self.coef_focal = coef_focal
        self.coef_dice = coef_dice

    def forward(self, outputs, targets):
        loss = 0.0

        for i in range(outputs.shape[1]):
            dice = self.weights[i]*self.dice_loss(outputs[:, i, ...], targets[:, i, ...])
            focal = self.weights[i]*self.focal_loss(outputs[:, i, ...], targets[:, i, ...])
            loss += self.coef_dice * dice + self.coef_focal * focal

        return loss

    
class DiceLoss(_Loss):

    def __init__(self, per_image=False):
        super(DiceLoss, self).__init__()
        self.per_image = per_image

    def forward(self, y_pred: Tensor, y_true: Tensor):
        """
        :param y_pred: NxCxHxW
        :param y_true: NxCxHxW
        :return: scalar
        """
        per_image = self.per_image
        y_pred = y_pred.sigmoid()
        
        batch_size = y_pred.size()[0]
        eps = 1e-5
        if not per_image:
            batch_size = 1
        
        dice_target = y_true.contiguous().view(batch_size, -1).float()
        dice_output = y_pred.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
        loss = (1 - (2 * intersection + eps) / union).mean()

        return loss
    
    
class FocalLoss(_Loss):

    def __init__(self, ignore_index=255, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index


    def forward(self, y_pred: Tensor, y_true: Tensor):
        """
        :param y_pred: NxCxHxW
        :param y_true: NxCxHxW
        :return: scalar
        """
        y_pred = y_pred.sigmoid()
        gamma = self.gamma
        ignore_index = self.ignore_index

        outputs = y_pred.contiguous()
        targets = y_true.contiguous()
        eps = 1e-8
        non_ignored = targets.view(-1) != ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** gamma * torch.log(pt)).mean()
