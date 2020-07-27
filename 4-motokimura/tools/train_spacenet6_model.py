#!/usr/bin/env python3
import os
import segmentation_models_pytorch as smp
import timeit
import torch

from tensorboardX import SummaryWriter

from _init_path import init_path
init_path()

from spacenet6_model.configs import load_config
from spacenet6_model.datasets import get_dataloader
from spacenet6_model.evaluations import get_metrics
from spacenet6_model.models import get_model
from spacenet6_model.solvers import (
    get_loss, get_lr_scheduler, get_optimizer
)
from spacenet6_model.utils import (
    checkpoint_epoch_filename, checkpoint_latest_filename,
    config_filename, dump_git_info, experiment_subdir, git_filename,
    load_latest_checkpoint, save_checkpoint,
    weight_best_filename
)


def main():
    """
    """

    config = load_config()
    print('successfully loaded config:')
    print(config)
    print('')

    epoch_to_start_val = config.EVAL.EPOCH_TO_START_VAL
    assert config.SOLVER.EPOCHS > epoch_to_start_val

    # prepare directories to output log/weight files
    exp_subdir = experiment_subdir(config.EXP_ID)
    log_dir = os.path.join(config.LOG_ROOT, exp_subdir)
    weight_dir = os.path.join(config.WEIGHT_ROOT, exp_subdir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)

    checkpoint_dir = os.path.join(config.CHECKPOINT_ROOT, exp_subdir)
    if config.SAVE_CHECKPOINTS:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # prepare dataloaders
    train_dataloader = get_dataloader(config, is_train=True)
    val_dataloader = get_dataloader(config, is_train=False)

    # prepare model to train
    model = get_model(config)

    # prepare optimizer with lr scheduler
    optimizer = get_optimizer(config, model)
    lr_scheduler = get_lr_scheduler(config, optimizer)

    # prepare other states
    start_epoch = 0
    best_score = 0

    # load checkpoint if exists
    model, optimizer, lr_scheduler, start_epoch, best_score = \
    load_latest_checkpoint(
        checkpoint_dir,
        model,
        optimizer,
        lr_scheduler,
        start_epoch,
        best_score
    )

    # prepare metrics and loss
    metrics = get_metrics(config)
    loss = get_loss(config)

    # prepare train/val epoch runners
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=config.MODEL.DEVICE,
        verbose=True,
    )
    val_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=config.MODEL.DEVICE,
        verbose=True,
    )

    # prepare tensorboard
    tblogger = SummaryWriter(log_dir)

    if config.DUMP_GIT_INFO:
        # save git hash
        dump_git_info(os.path.join(log_dir, git_filename()))

    # dump config to a file
    with open(os.path.join(log_dir, config_filename()), 'w') as f:
        f.write(str(config))

    # train loop
    metric_name = config.EVAL.MAIN_METRIC
    split_id = config.INPUT.TRAIN_VAL_SPLIT_ID

    for epoch in range(start_epoch, config.SOLVER.EPOCHS):
        lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch: {epoch}, lr: {lr}')

        # run train for 1 epoch
        train_logs = train_epoch.run(train_dataloader)

        # log lr to tensorboard
        tblogger.add_scalar('lr', lr, epoch)
        # log train losses and scores
        for k, v in train_logs.items():
            tblogger.add_scalar(f'split_{split_id}/train/{k}', v, epoch)

        if epoch >= config.EVAL.EPOCH_TO_START_VAL:
            # run val for 1 epoch
            val_logs = val_epoch.run(val_dataloader)

            # log val losses and scores
            for k, v in val_logs.items():
                tblogger.add_scalar(f'split_{split_id}/val/{k}', v, epoch)

            # save model weight if score updated
            if best_score < val_logs[metric_name]:
                best_score = val_logs[metric_name]
                torch.save(
                    model.state_dict(),
                    os.path.join(weight_dir, weight_best_filename())
                )
                print('Best val score updated!')
        else:
            print(f'Skip val until epoch {epoch_to_start_val}')

        # update lr for the next epoch
        lr_scheduler.step()

        if config.SAVE_CHECKPOINTS:
            # save checkpoint every epoch
            save_checkpoint(
                os.path.join(checkpoint_dir, checkpoint_epoch_filename(epoch)),
                model, optimizer, lr_scheduler, epoch + 1, best_score)
            save_checkpoint(
                os.path.join(checkpoint_dir, checkpoint_latest_filename()),
                model, optimizer, lr_scheduler, epoch + 1, best_score)

    tblogger.close()


if __name__ == '__main__':
    t0 = timeit.default_timer()

    main()

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60.0))
