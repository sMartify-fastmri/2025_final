
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import time
from pathlib import Path
import copy

from collections import defaultdict

from moe_utils_rescue.learning.classifier_part import call_classifier, make_classifier
from moe_utils_rescue.data.load_data_withclass import create_data_loaders_withclass
from moe_utils_rescue.common.scheduler import CosineAnnealingWarmupRestarts

from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss, mask_based_loss
from moe_utils_rescue.model.moe_wrapper import MixtureOfPromptMR

from utils.data.mraugment.data_augment import DataAugmentor

import os


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')

    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    if args.regularizer_loss is None:
        loss_reg = None
    elif args.regularizer_loss == 'L1':
        loss_reg = nn.L1Loss().to(device=device)
    elif args.regularizer_loss == 'L2':
        loss_reg = nn.MSELoss().to(device=device)

    iter_class = 0
    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _, gamyun, gamyun_area_ratio, class_label = data
        # RESCUE HELPER!! IGNORE BRAIN OR KNEE DATA TO REDUCE TIME CONSUMPTION
        if (class_label.argmax().item() < 2) and (args.ignore_brain_or_knee == 'brain'):
            if iter % args.report_interval == 0:
                print(
                    f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                    f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                    f'Pass this iteration: brain detected '
                    f'(agg) = {total_loss/(iter_class+1):01.5f} | '
                    f'Time = {time.perf_counter() - start_iter:.4f}s',
                )
                start_iter = time.perf_counter()
            continue
        elif (class_label.argmax().item() > 1) and (args.ignore_brain_or_knee == 'knee'):
            if iter % args.report_interval == 0:
                print(
                    f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                    f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                    f'Pass this iteration: knee detected '
                    f'(agg) = {total_loss/(iter_class+1):01.5f} | '
                    f'Time = {time.perf_counter() - start_iter:.4f}s',
                )
                start_iter = time.perf_counter()
            continue

        iter_class += 1
        
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        gamyun = gamyun.cuda(non_blocking=True).float()
        class_label = class_label.cuda(non_blocking=True)
        
        mask_based_weight = mask_based_loss(gamyun_area_ratio) if args.gamyun_loss else 1.
        mask_based_weight = mask_based_weight.cuda(non_blocking=True)
        gamyun[gamyun == 0] = args.gamyun_shade_strength if epoch >= args.gamyun_start_epoch + args.starting_epoch - 1 else 1.
        
        optimizer.zero_grad()
        output = model(kspace, mask, class_label, use_checkpoint=args.use_checkpoint)

        loss = loss_type(output  * gamyun, target * gamyun, maximum) * mask_based_weight

        if loss_reg is not None and epoch >= args.regularizer_start_epoch + args.starting_epoch - 1:
            reg_term = args.regularizer_lambda * loss_reg(output * gamyun, target * gamyun) * mask_based_weight
            loss = loss + reg_term
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():01.5f} '
                f'(agg) = {total_loss/(iter_class+1):01.5f} | '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
        
    total_loss = total_loss / (iter_class + 1)
    return total_loss, time.perf_counter() - start_epoch

def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    gamyuns = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices, gamyun, _, class_label = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask, class_label)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()
                gamyuns[fnames[i]][int(slices[i])] = gamyun[i].numpy()
                
    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    for fname in gamyuns:
        gamyuns[fname] = np.stack(
            [out for _, out in sorted(gamyuns[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname] * gamyuns[fname], reconstructions[fname] * gamyuns[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, scheduler, best_val_loss, is_new_best):
    for i, class_name in enumerate(['brain_acc4', 'brain_acc8', 'knee_acc4', 'knee_acc8']):
        model_name = f'model_epoch{epoch}in{args.num_epochs}_{class_name}.pt' if args.save_each_epoch else f'model_recent_{class_name}.pt'
        submodel = model.experts[i]
        torch.save(
            {
                'epoch': epoch,
                'args': args,
                'model': submodel.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'exp_dir': exp_dir
            },
            f=exp_dir / model_name
        )
        print(f'Model saved: {model_name} in {str(exp_dir)}')
        if is_new_best:
            shutil.copyfile(exp_dir / model_name, exp_dir / f'best_model_{class_name}.pt')


def call_model(args):
    # Load current model, or another name of the model.
    start_exp_dir = '../result' / args.starting_expert / 'checkpoints'
    file_path = start_exp_dir / f'model_epoch{args.starting_epoch}in{args.starting_expert_max_epoch}.pt'
    checkpoints = torch.load(file_path, map_location='cpu', weights_only=False)
    
    return checkpoints, file_path

def call_each_model(args):
    if args.call_moe_units:
        all_checkpoints = []

        call_exp_dir = '../result' / args.call_moe_name / 'checkpoints'
        epoch2call_forclass = [args.brain_acc4_epoch2call, args.brain_acc8_epoch2call, args.knee_acc4_epoch2call, args.knee_acc8_epoch2call]
        for i, class_name in enumerate(['brain_acc4', 'brain_acc8', 'knee_acc4', 'knee_acc8']):
            model_name = f'model_epoch{epoch2call_forclass[i]}in{args.max_epoch2call}_{class_name}.pt'
            file_path = call_exp_dir / model_name

            checkpoints = torch.load(file_path, map_location='cpu', weights_only=False)
            all_checkpoints.append(checkpoints)

    return all_checkpoints   

def train(args):

    # [DEVICE SETUP]
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # [CLASSIFIER CALL]
    try:
        classifier = call_classifier(args)
    except:
        classifier = make_classifier(args)
    classifier.to(device=device)

    # [MODEL CALL]
    checkpoints, called_path = call_model(args)
    model = MixtureOfPromptMR(args, checkpoints=checkpoints['model'])
    model.to(device=device)

    if args.call_moe_units:
        all_checkpoints = call_each_model(args)
        model.load_each_state_dict(*all_checkpoints)

    # [LOSS]
    loss_type = SSIMLoss().to(device=device)
    best_val_loss = checkpoints['best_val_loss'].item()

    # [OPTIMIZER]
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # [SCHEDULER]
    if args.lr_scheduler == 'StepLR':
        warmup_steps = min(args.brain_aug_delay, args.knee_aug_delay)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        const_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.scheduler_start_epoch - warmup_steps)
        step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler_step_size, args.scheduler_gamma)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, const_scheduler, step_scheduler],
            milestones=[warmup_steps, args.scheduler_start_epoch]
        )
    elif args.lr_scheduler == 'CosineAnnealingWarmupRestarts':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer=optimizer,
            first_cycle_steps=args.scheduler_first_cycle_epochs,
            cycle_mult=args.scheduler_cycle_mult,
            max_lr=args.lr,
            min_lr=args.scheduler_lr_min,
            warmup_steps=args.scheduler_warmup_epochs,
            gamma=args.scheduler_cosine_gamma,
            last_epoch=args.num_epochs - args.starting_epoch - 1
        )

    # [AUGMENTATION]
    args_aug_brain = copy.deepcopy(args)
    args_aug_brain.max_epochs = args.num_epochs - args.annealing_epoch
    assert args_aug_brain.max_epochs >= args.starting_epoch, "'--annealing-epoch' greater than '--num-epochs'"
    args_aug_brain.aug_delay = args.brain_aug_delay + args.starting_epoch
    args_aug_brain.aug_strength = args.brain_aug_strength
    args_aug_brain.aug_weight_translation = args.brain_aug_weight_translation
    args_aug_brain.aug_weight_rotation = args.brain_aug_weight_rotation
    args_aug_brain.aug_weight_shearing = args.brain_aug_weight_shearing
    args_aug_brain.aug_weight_scaling = args.brain_aug_weight_scaling
    args_aug_brain.aug_weight_rot90 = args.brain_aug_weight_rot90
    args_aug_brain.aug_weight_fliph = args.brain_aug_weight_fliph
    args_aug_brain.aug_weight_flipv = args.brain_aug_weight_flipv
    args_aug_brain.aug_max_translation_x = args.brain_aug_max_translation_x
    args_aug_brain.aug_max_translation_y = args.brain_aug_max_translation_y
    args_aug_brain.aug_max_rotation = args.brain_aug_max_rotation
    args_aug_brain.aug_max_shearing_x = args.brain_aug_max_shearing_x
    args_aug_brain.aug_max_shearing_y = args.brain_aug_max_shearing_y
    args_aug_brain.aug_max_scaling = args.brain_aug_max_scaling
    args_aug_brain.aug_on = True

    args_aug_knee = copy.deepcopy(args)
    args_aug_knee.max_epochs = args.num_epochs - args.annealing_epoch
    assert args_aug_knee.max_epochs >= args.starting_epoch, "'--annealing-epoch' greater than '--num-epochs'"
    args_aug_knee.aug_delay = args.knee_aug_delay + args.starting_epoch
    args_aug_knee.aug_strength = args.knee_aug_strength
    args_aug_knee.aug_weight_translation = args.knee_aug_weight_translation
    args_aug_knee.aug_weight_rotation = args.knee_aug_weight_rotation
    args_aug_knee.aug_weight_shearing = args.knee_aug_weight_shearing
    args_aug_knee.aug_weight_scaling = args.knee_aug_weight_scaling
    args_aug_knee.aug_weight_rot90 = args.knee_aug_weight_rot90
    args_aug_knee.aug_weight_fliph = args.knee_aug_weight_fliph
    args_aug_knee.aug_weight_flipv = args.knee_aug_weight_flipv
    args_aug_knee.aug_max_translation_x = args.knee_aug_max_translation_x
    args_aug_knee.aug_max_translation_y = args.knee_aug_max_translation_y
    args_aug_knee.aug_max_rotation = args.knee_aug_max_rotation
    args_aug_knee.aug_max_shearing_x = args.knee_aug_max_shearing_x
    args_aug_knee.aug_max_shearing_y = args.knee_aug_max_shearing_y
    args_aug_knee.aug_max_scaling = args.knee_aug_max_scaling
    args_aug_knee.aug_on = True

    current_epoch = [args.starting_epoch]
    current_epoch_fn = lambda: current_epoch[0]

    augmentor_brain = DataAugmentor(args_aug_brain, current_epoch_fn, seed=args.seed)
    augmentor_knee = DataAugmentor(args_aug_knee, current_epoch_fn, seed=args.seed)

    if augmentor_brain.aug_on and augmentor_knee.aug_on:
        augmentor_brain.seed_pipeline(args.seed + 2000)
        augmentor_knee.seed_pipeline(args.seed + 2000)

    # [DATALOADER]
    train_loader = create_data_loaders_withclass(
        data_path=args.data_path_train, 
        args=args, 
        classifier=classifier,
        shuffle=True, 
        brain_augmentor=augmentor_brain, 
        knee_augmentor=augmentor_knee
    )
    val_loader = create_data_loaders_withclass(
        data_path=args.data_path_val, 
        args=args, 
        classifier=classifier,
        shuffle=args.use_val_for_final, 
        brain_augmentor=(augmentor_brain if args.use_val_for_final else None), 
        knee_augmentor=(augmentor_knee if args.use_val_for_final else None)
    )


    file_path_load = os.path.join(args.val_loss_dir, 'val_loss_log.npy')
    try:
        val_loss_log = np.load(file_path_load)
    except OSError:
        print('Starting from the very first of MoE training... validation loss log is empty.')
        val_loss_log = np.empty((0, 2))


    for epoch in range(args.starting_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} from pretrained model at epoch {args.starting_epoch} ............... {args.net_name} ...............')

        if augmentor_brain is not None and epoch < args_aug_brain.max_epochs:
            current_epoch[0] = epoch
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)

        if args.use_val_for_final:
            assert args.save_each_epoch == True, "If '--use-val-for-final', '--save-each-epoch' should be True."
            val_loss, val_time = train_epoch(args, epoch, model, val_loader, optimizer, loss_type)
        else:
            val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
            val_loss = val_loss / num_subjects
        
        scheduler.step()
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, 'val_loss_log')
        np.save(file_path, val_loss_log)
        print(f'loss file saved! {file_path}')

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, scheduler, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best and not args.use_val_for_final:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )