import argparse
import os
import random

import cv2

import transforms
from spacenet_dataset import SpacenetLocDataset
from tools.metrics import calculate_metrics

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import zoo

from albumentations import Compose, PadIfNeeded, RandomBrightnessContrast, \
    RandomGamma, RandomRotate90, IAAAdditiveGaussianNoise, MedianBlur, DualTransform, HorizontalFlip, Resize

import losses

from apex.parallel import DistributedDataParallel, convert_syncbn_model
from tensorboardX import SummaryWriter

from tools.config import load_config
from tools.utils import create_optimizer, AverageMeter

from apex import amp

from losses import dice_round

import numpy as np
import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist

torch.backends.cudnn.benchmark = True


def create_val_transforms(conf):
    return Compose([
        PadIfNeeded(min_height=928, min_width=928)
    ])


def main():
    parser = argparse.ArgumentParser("PyTorch Spacenet6 Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file')
    arg('--workers', type=int, default=8, help='number of cpu threads to use')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--prefix', type=str, default='localization_')
    arg('--data-dir', type=str, default="/mnt/sota/datasets/spacenet/train/AOI_11_Rotterdam/")
    arg('--folds-csv', type=str, default='folds.csv')
    arg('--logdir', type=str, default='logs')
    arg('--zero-score', action='store_true', default=False)
    arg('--from-zero', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg("--local_rank", default=0, type=int)
    arg("--opt-level", default='O1', type=str)
    arg("--predictions", default="/wdata/oof_preds", type=str)
    arg("--test_every", type=int, default=1)
    arg("--visualizer-path", type=str, default="visualizer/visualizer.jar")

    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True

    conf = load_config(args.config)
    model = zoo.__dict__[conf['network']](seg_classes=conf['num_classes'], backbone_arch=conf['encoder'])

    model = model.cuda()
    if args.distributed:
        model = convert_syncbn_model(model)
    mask_loss_function = losses.__dict__[conf["mask_loss"]["type"]](**conf["mask_loss"]["params"]).cuda()
    loss_functions = {"mask_loss": mask_loss_function}
    optimizer, scheduler = create_optimizer(conf['optimizer'], model)

    dice_best = 0
    f_score_best = 0
    start_epoch = 0
    batch_size = conf['optimizer']['batch_size']

    data_train = SpacenetLocDataset(mode="train",
                                    fold=args.fold,
                                    data_path=args.data_dir,
                                    folds_csv=args.folds_csv,
                                    transforms=transforms.__dict__[conf["transforms"]](conf["input"]),
                                    multiplier=conf["data_multiplier"])
    data_val = SpacenetLocDataset(mode="val",
                                  fold=args.fold,
                                  data_path=args.data_dir,
                                  folds_csv=args.folds_csv,
                                  transforms=create_val_transforms(conf['input'])
                                  )
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data_train)

    train_data_loader = DataLoader(data_train, batch_size=batch_size, num_workers=args.workers,
                                   shuffle=train_sampler is None, sampler=train_sampler, pin_memory=False,
                                   drop_last=True)
    val_batch_size = 1
    val_data_loader = DataLoader(data_val, batch_size=val_batch_size, num_workers=args.workers, shuffle=False,
                                 pin_memory=False)
    snapshot_name = "{}{}_{}_{}".format(args.prefix, conf['network'], conf['encoder'], args.fold)

    os.makedirs(args.logdir, exist_ok=True)
    summary_writer = SummaryWriter(args.logdir + '/' + snapshot_name)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            state_dict = checkpoint['state_dict']
            if conf['optimizer'].get('zero_decoder', False):
                for key in state_dict.copy().keys():
                    if key.startswith("module.final"):
                        del state_dict[key]
            state_dict = {k[7:]: w for k, w in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            if not args.from_zero:
                start_epoch = checkpoint['epoch']
                if not args.zero_score:
                    dice_best = checkpoint.get('dice_best', 0)
                    f_score_best = checkpoint.get('f_score_best', 0)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.from_zero:
        start_epoch = 0
    current_epoch = start_epoch

    if conf['fp16']:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.opt_level,
                                          loss_scale='dynamic')


    if args.distributed:
        model = DistributedDataParallel(model, delay_allreduce=True)
    else:
        model = DataParallel(model).cuda()
    for epoch in range(start_epoch, conf['optimizer']['schedule']['epochs']):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.module.train()
        train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                    args.local_rank)

        model = model.eval()
        if args.local_rank == 0:
            torch.save({
                'epoch': current_epoch + 1,
                'state_dict': model.state_dict(),
                'dice_best': dice_best,
                'f_score_best': f_score_best,
            }, args.output_dir + '/' + snapshot_name + "_last")
            if epoch % args.test_every == 0:
                preds_dir = os.path.join(args.predictions, snapshot_name)
                dice_best, f_score_best = evaluate_val(args, val_data_loader, dice_best, f_score_best,
                                         model,
                                         snapshot_name=snapshot_name,
                                         current_epoch=current_epoch,
                                         optimizer=optimizer, summary_writer=summary_writer,
                                         predictions_dir=preds_dir, data_dir=args.data_dir, visualizer_path=args.visualizer_path)
        current_epoch += 1


def evaluate_val(args, data_val, dice_best, f_score_best, model, snapshot_name, current_epoch, optimizer, summary_writer,
                 predictions_dir, data_dir, visualizer_path):
    print("Test phase")
    model = model.eval()
    dice, f_score = validate(model, data_loader=data_val, predictions_dir=predictions_dir, data_dir=data_dir,
                             visualizer_path=visualizer_path)
    if args.local_rank == 0:
        summary_writer.add_scalar('val/dice', float(dice), global_step=current_epoch)
        summary_writer.add_scalar('val/F1', f_score, global_step=current_epoch)
        if dice > dice_best:
            print("Dice improved from {} to {}".format(dice_best, dice))
            if args.output_dir is not None:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'dice_best': dice,
                    'dice': dice,
                    'f_score': f_score,
                    'f_score_best': f_score_best,
                }, args.output_dir + snapshot_name + "_best_dice")
            dice_best = dice
        if f_score > f_score_best:
            print("F1 improved from {} to {}".format(f_score_best, f_score))
            if args.output_dir is not None:
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': model.state_dict(),
                    'dice_best': dice_best,
                    'dice': dice,
                    'f_score': f_score,
                    'f_score_best': f_score,
                }, args.output_dir + snapshot_name + "_best_f1")
            f_score_best = f_score
        torch.save({
            'epoch': current_epoch + 1,
            'state_dict': model.state_dict(),
            'dice_best': dice_best,
            'f_score_best': f_score_best,
            'dice': dice,
            'f_score': f_score,
        }, args.output_dir + snapshot_name + "_last")
        print("dice: {}, dice_best: {}".format(dice, dice_best))
    return dice_best, f_score_best


def validate(net, data_loader, predictions_dir, data_dir, visualizer_path):
    os.makedirs(predictions_dir, exist_ok=True)
    preds_dir = predictions_dir + "/predictions"
    os.makedirs(preds_dir, exist_ok=True)
    dices = []
    with torch.no_grad():
        for sample in tqdm(data_loader):
            imgs = sample["image"].cuda().float()
            mask = sample["mask"].cuda().float()

            output = net(imgs)
            binary_pred = torch.sigmoid(output)

            for i in range(output.shape[0]):
                d = dice_round(binary_pred[:, 0:1, :], mask[:, 0:1, ...], t=0.5).item()
                dices.append(d)
                cv2.imwrite(os.path.join(preds_dir, sample["img_name"][i] + ".png"),
                            (np.moveaxis(binary_pred[i].cpu().numpy(), 0, -1)[..., :3] * 255))
    f_score = calculate_metrics(fold_dir=predictions_dir,
                                visualizer_path=visualizer_path,
                                truth_csv=os.path.join(data_dir,
                                                       "SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv"),
                                img_dir=os.path.join(data_dir, "SAR-Intensity/"),
                                sar_orientations_csv=os.path.join(data_dir, "SummaryData/SAR_orientations.txt"))
    return np.mean(dices), f_score


def train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf,
                local_rank):
    losses = AverageMeter()
    dices = AverageMeter()
    iterator = tqdm(train_data_loader)
    model.train()
    if conf["optimizer"]["schedule"]["mode"] == "epoch":
        scheduler.step(current_epoch)
    for i, sample in enumerate(iterator):
        imgs = sample["image"].cuda()
        masks = sample["mask"].cuda().float()
        out_mask = model(imgs)
        with torch.no_grad():
            pred = torch.sigmoid(out_mask)
            d = dice_round(pred[:, 0:1, ...], masks[:, 0:1, ...], t=0.5).item()
        dices.update(d, imgs.size(0))

        mask_loss = loss_functions["mask_loss"](out_mask, masks)
        loss = mask_loss
        losses.update(loss.item(), imgs.size(0))
        iterator.set_description(
            "epoch: {}; lr {:.7f}; Loss ({loss.avg:.4f}); dice ({dice.avg:.4f}); ".format(
                current_epoch, scheduler.get_lr()[-1], loss=losses, dice=dices))
        optimizer.zero_grad()
        if conf['fp16']:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
        optimizer.step()
        torch.cuda.synchronize()

        if conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
            scheduler.step(i + current_epoch * len(train_data_loader))

    if local_rank == 0:
        for idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            summary_writer.add_scalar('group{}/lr'.format(idx), float(lr), global_step=current_epoch)
        summary_writer.add_scalar('train/loss', float(losses.avg), global_step=current_epoch)


if __name__ == '__main__':
    main()
