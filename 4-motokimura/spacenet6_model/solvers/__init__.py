import segmentation_models_pytorch as smp
import torch.optim

from .losses import CombinedLoss, BinaryFocalLoss


def get_optimizer(config, model):
    """
    """
    optimizer_name = config.SOLVER.OPTIMIZER
    if optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.SOLVER.LR,
            weight_decay=config.SOLVER.WEIGHT_DECAY
        )
    elif optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.SOLVER.LR,
            weight_decay=config.SOLVER.WEIGHT_DECAY
        )
    else:
        raise ValueError()


def get_lr_scheduler(config, optimizer):
    """
    """
    scheduler_name = config.SOLVER.LR_SCHEDULER
    if scheduler_name == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.SOLVER.LR_MULTISTEP_MILESTONES,
            gamma=config.SOLVER.LR_MULTISTEP_GAMMA
        )
    elif scheduler_name == 'annealing':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.SOLVER.LR_ANNEALING_T_MAX,
            eta_min=config.SOLVER.LR_ANNEALING_ETA_MIN,
        )
    else:
        raise ValueError()


def get_loss(config):
    """
    """
    def _get_loss(config, loss_name):
        if loss_name == 'bce':
            return smp.utils.losses.BCELoss()
        elif loss_name == 'dice':
            return smp.utils.losses.DiceLoss()
        elif loss_name == 'focal':
            return BinaryFocalLoss(
                gamma=config.SOLVER.FOCAL_LOSS_GAMMA
            )
        else:
            raise ValueError()

    loss_modules = []
    for loss_name in config.SOLVER.LOSSES:
        loss_modules.append(_get_loss(config, loss_name))

    return CombinedLoss(
        loss_modules,
        config.SOLVER.LOSS_WEIGHTS
    )
