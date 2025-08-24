import torch
import torch.nn as nn
import argparse
import copy
from pathlib import Path

from utils.common.utils import seed_fix

from moe_utils.model.resnet import ResNet50

# Requires args.batch_size
from moe_utils.data.load_data_for_classifier import create_classifier_data_loader

# Requires args.classifier_name, classifier_path
from moe_utils.learning.classifier_part import save_classifier 


def parse():
    parser = argparse.ArgumentParser(description="Make awesome brain/knee classifier!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--classifier-name', type=Path, default='awesome_classifier')
    parser.add_argument('--classifier-path', type=Path, default='../result/classifier/')
    parser.add_argument('--classifier-epoch', type=int, default=10)

    # Inverted data path for proper validation of classifier
    parser.add_argument('--classifier-data-path-train', type=Path, default='../Data/val/')
    parser.add_argument('--classifier-data-path-val', type=Path, default='../Data/train/')

    parser.add_argument('--lr', type=float, default=2e-4)

    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()

    if args.seed is not None:
        seed_fix(args.seed)

    args.classifier_checkpoint_path = args.classifier_path / 'checkpoints'
    args.classifier_checkpoint_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('[Training awesome classifier...] Current cuda device: ', torch.cuda.current_device())

    model = ResNet50()
    model.to(device=device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.01*args.lr)

    val_args = copy.deepcopy(args)
    val_args.batch_size = 1

    train_loader = create_classifier_data_loader(args.classifier_data_path_train, args, shuffle=True, isforward=False)
    val_loader = create_classifier_data_loader(args.classifier_data_path_val, val_args, shuffle=False, isforward=False)
    train_loader_correction = create_classifier_data_loader(args.classifier_data_path_train, val_args, shuffle=False, isforward=False)

    best_val_loss = 1.

    for epoch in range(args.classifier_epoch):
        train_loss = 0

        model.train()
        for iter, data in enumerate(train_loader):
            image_grappa, target = data
            image_grappa = image_grappa.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(image_grappa.unsqueeze(1))
            loss = loss_fn(output, target)
            
            l2_reg = torch.tensor(0., device=device)
            for p in model.parameters():
                l2_reg += torch.norm(p, 2)**2
            loss = loss + 1e-6 * l2_reg
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f'[Training classifier...] Epoch {epoch} | Training loss: {train_loss}')

        model.eval()

        with torch.no_grad():
            val_loss, incorrect_count = 0, 0
            for iter, data in enumerate(val_loader):
                image_grappa, target = data
                image_grappa = image_grappa.to(device)
                target = target.to(device)
                output = model(image_grappa.unsqueeze(0))
                loss = loss_fn(output, target)
                val_loss += loss.item()
                if torch.argmax(output) != torch.argmax(target):
                    incorrect_count += 1
    
        val_loss /= len(val_loader)
        print(f'[Validating classifier...] Validation loss: {val_loss}, Incorrect count: {incorrect_count}/{len(val_loader)}')

        if val_loss < best_val_loss:
            print('@@@@@@@@@@@@@@@@@@@@@@@ NewRecord @@@@@@@@@@@@@@@@@@@@@@@')
            print(f'Incorrect in val set: {incorrect_count}')

            best_val_loss = val_loss

            with torch.no_grad():
                val_loss, incorrect_count = 0, 0
                for iter, data in enumerate(train_loader_correction):
                    image_grappa, target = data
                    image_grappa = image_grappa.to(device)
                    target = target.to(device)
                    output = model(image_grappa.unsqueeze(0))
                    loss = loss_fn(output, target)
                    val_loss += loss.item()
                    if torch.argmax(output) != torch.argmax(target):
                        incorrect_count += 1
            print(f'Incorrect in train set: {incorrect_count}')

            save_classifier(model, args)

        scheduler.step()
            
            
        
    