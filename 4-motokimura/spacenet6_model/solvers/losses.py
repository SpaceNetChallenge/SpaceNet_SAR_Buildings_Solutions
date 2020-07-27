import segmentation_models_pytorch as smp
import torch


class CombinedLoss(smp.utils.base.Loss):
    """
    """
    __name__ = 'loss'

    def __init__(
        self,
        loss_modules,
        loss_weights,
        **kwargs
    ):
        assert len(loss_modules) == len(loss_weights)

        super().__init__(**kwargs)

        self._loss_modules = loss_modules
        self._loss_weights = loss_weights

    def forward(self, y_pr, y_gt):
        """
        """
        losses = self._loss_modules
        weights = self._loss_weights

        loss = losses[0](y_pr, y_gt) * weights[0]
        if len(losses) == 1:
            return loss

        for i in range(1, len(losses)):
            loss += losses[i](y_pr, y_gt) * weights[i]
        return loss


class BinaryFocalLoss(smp.utils.base.Loss):
    """
    """
    # references: 
    # https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions/blob/812f151d244565f29987ebec7683ef42622ae16e/cannab/losses.py#L259
    def __init__(
        self,
        gamma=2.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self._gamma = gamma

    def forward(self, y_pr, y_gt):
        """
        """
        eps = 1e-8
        y_gt = torch.clamp(y_gt, eps, 1 - eps)
        y_pr = torch.clamp(y_pr, eps, 1 - eps)

        pt = (1 - y_gt) * (1 - y_pr) + y_gt * y_pr
        loss = - (1 - pt) ** self._gamma * torch.log(pt)
        return loss.mean()
