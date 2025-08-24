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
from utils.data.load_data import create_data_loaders
from utils.data.generate_image_mask import generate
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss, mask_based_loss
from utils.model.promptmr_plus import PromptMR

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

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, _, _, gamyun, gamyun_area_ratio = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)
        gamyun = gamyun.cuda(non_blocking=True).float()

        mask_based_weight = mask_based_loss(gamyun_area_ratio) if args.gamyun_loss else 1.
        mask_based_weight = mask_based_weight.cuda(non_blocking=True)
        gamyun[gamyun == 0] = args.gamyun_shade_strength if epoch >= args.gamyun_start_epoch else 1.

        optimizer.zero_grad()
        output = model(kspace, mask, use_checkpoint=args.use_checkpoint)

        loss = loss_type(output * gamyun, target * gamyun, maximum) * mask_based_weight

        if loss_reg is not None and epoch >= args.regularizer_start_epoch:
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
                f'(agg) = {total_loss/(iter+1):01.5f} | '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    
    
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    gamyuns = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices, gamyun, _ = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

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
    model_name = f'model_epoch{epoch}in{args.num_epochs}.pt' if args.save_each_epoch else 'model_recent.pt'
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / model_name
    )
    print(f'Model saved: {model_name} in {str(exp_dir)}')
    if is_new_best:
        shutil.copyfile(exp_dir / model_name, exp_dir / 'best_model.pt')


def call_model(args):
    checkpoints, file_path = None, None
    try:
        # Load current model, or another name of the model.
        start_exp_dir = args.exp_dir if args.start_from_net is None else ('../result' / args.start_from_net / 'checkpoints')
        if args.start_from_best:
            file_path = start_exp_dir / 'best_model.pt'
        elif args.start_from_epoch > 0:
            file_path = start_exp_dir / f'model_epoch{args.start_from_epoch}in{args.num_epochs}.pt'
        else:
            assert (
                args.start_from_net is None
            ), f"If '--start-from-epoch' = {args.start_from_epoch}, '--start-from-net' should be None."

        if file_path is not None:
            checkpoints = torch.load(file_path, map_location='cpu', weights_only=False)
    
    except Exception as e:
        print(f'[WARNING] Model call failed; starting from the first epoch.\nError message: {e}')

    load_prev = checkpoints is not None

    return checkpoints, load_prev, file_path

        
def train(args):

    # Generate image masks before training
    generate(args, train_dir=args.data_path_train, val_dir=args.data_path_val)
    
    # [DEVICE SETUP]
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # [MODEL CALL]
    model = PromptMR(
        num_cascades=args.num_cascades,
        num_adj_slices=1,   # TODO Parameters considering adjacent slices are not implemented.
        n_feat0=args.n_feat0,
        feature_dim=args.feature_dim,
        prompt_dim=args.prompt_dim,
        sens_n_feat0=args.sens_n_feat0,
        sens_feature_dim=args.sens_feature_dim,
        sens_prompt_dim=args.sens_prompt_dim,
        len_prompt=args.len_prompt,
        prompt_size=args.prompt_size,
        n_enc_cab=args.n_enc_cab,
        n_dec_cab=args.n_dec_cab,
        n_skip_cab=args.n_skip_cab,
        n_bottleneck_cab=args.n_bottleneck_cab,
        no_use_ca=args.no_use_ca,
        adaptive_input=args.adaptive_input,
        n_buffer=args.n_buffer,
        n_history=args.n_history,
        use_sens_adj=args.use_sens_adj
    )
    model.to(device=device)


    # [PREV MODEL CALL IF NEEDED]
    checkpoints, load_prev, called_path = call_model(args)

    if load_prev:
        try:
            model.load_state_dict(checkpoints['model'])
            model_instruction = 'The same model' if args.start_from_net is None else f"Model '{args.start_from_net}'"
            best_instruction = ', which is the best of the model' if args.start_from_best else ''
            print(f"{model_instruction} called at epoch {checkpoints['epoch']}: val_loss = {checkpoints['best_val_loss'].item()}{best_instruction}.")
        except Exception as e:
            print(f'[WARNING] Model architecture seems not to be matched; starting with a new model...\nError message: {e}')
            
    best_val_loss = checkpoints['best_val_loss'].item() if load_prev else 1.
    start_epoch = checkpoints['epoch'] if load_prev else 0
    

    # [LOSS]
    loss_type = SSIMLoss().to(device=device)

    # [OPTIMIZER]
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'RAdam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    if load_prev and args.call_optimizer:
        optimizer.load_state_dict(checkpoints['optimizer'])

    # [SCHEDULER]
    if args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler_step_size, args.scheduler_gamma)
    # TODO: Add other schedulers

    if load_prev and args.call_scheduler:
        scheduler.load_state_dict(checkpoints['scheduler'])

    # [AUGMENTATION]
    args_aug = argparse.Namespace(**vars(args))
    args_aug.max_epochs = args.num_epochs - args.annealing_epoch
    assert args_aug.max_epochs >= 0, "'--annealing-epoch' greater than '--num-epochs'"
    args_aug.aug_on = True

    current_epoch = [start_epoch]
    current_epoch_fn = lambda: current_epoch[0]
    # Pass the seed to the augmentor for reproducible augmentations
    augmentor = DataAugmentor(args_aug, current_epoch_fn, seed=args.seed)
    
    # Additional seeding for the augmentation pipeline
    if augmentor.aug_on:
        augmentor.seed_pipeline(args.seed + 2000)

    # [DATALOADER]
    train_loader = create_data_loaders(data_path=args.data_path_train, args=args, shuffle=True, augmentor=augmentor)
    val_loader = create_data_loaders(data_path=args.data_path_val, args=args, shuffle=args.use_val_for_final, augmentor=(augmentor if args.use_val_for_final else None))

    # [LOADING VALIDATION LOSS LOG]
    if args.start_from_net is None:
        file_path_load = os.path.join(args.val_loss_dir, 'val_loss_log.npy')
    else:
        file_path_load = os.path.join('../result' / args.start_from_net, 'val_loss_log.npy')
    try:
        val_loss_log = np.load(file_path_load)
    except OSError:
        if start_epoch != 0:
            print(f"[WARNING] Loss file in '{args.val_loss_dir}' seems to be expired.")
        val_loss_log = np.empty((0, 2))
    


    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        if augmentor is not None and epoch < args_aug.max_epochs:
            current_epoch[0] = epoch

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)

        if args.use_val_for_final:
            assert args.save_each_epoch == True, "If '--use-val-for-final', '--save-each-epoch' should be True."
            val_loss, val_time = train_epoch(args, epoch, model, val_loader, optimizer, loss_type)
        else:
            val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
            val_loss = val_loss / num_subjects
        
        if epoch >= args.scheduler_start_epoch:
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
        
