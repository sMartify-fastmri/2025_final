import torch
import torch.nn as nn

from moe_utils_rescue.model.resnet import ResNet50
from moe_utils_rescue.data.load_data_for_classifier import create_classifier_data_loader


def call_classifier(args) -> nn.Module:
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('[Calling classifier...] Current cuda device: ', torch.cuda.current_device())

    file_path = args.classifier_path / 'checkpoints' / f'{args.classifier_name}.pt'
    checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)

    model = ResNet50()
    model.load_state_dict(checkpoint['model'])
    
    model.to(device=device)

    return model


def save_classifier(classifier, args):
    model_name = f'{args.classifier_name}.pt'
    torch.save(
        {
            'model': classifier.state_dict()
        },
        f=args.classifier_path / 'checkpoints' / model_name
    )
    print(f"Model saved: {model_name} in {str(args.classifier_path / 'checkpoints')}")


def make_classifier(args, save=True):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('[Training classifier...] Current cuda device: ', torch.cuda.current_device())

    model = ResNet50()
    model.to(device=device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = create_classifier_data_loader(args.classifier_data_path_train, args, shuffle=True, isforward=False)
    val_loader = create_classifier_data_loader(args.classifier_data_path_val, args, shuffle=args.use_val_for_final, isforward=False)

    for epoch in range(args.classifier_epoch):
        train_loss = 0
        for iter, data in enumerate(train_loader):
            image_grappa, target = data
            image_grappa = image_grappa.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(image_grappa.unsqueeze(0))
            loss = loss_fn(output, target)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f'[Training classifier...] Epoch {epoch} | Training loss: {train_loss}')

        if args.use_val_for_final:
            train_loss = 0
            for iter, data in enumerate(val_loader):
                image_grappa, target = data
                image_grappa = image_grappa.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(image_grappa.unsqueeze(0))
                loss = loss_fn(output, target)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(val_loader)
            print(f'[Training classifier for val set...] Epoch {epoch} | Training loss: {train_loss}')
        else:
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
    
    model.eval()

    if save:
        save_classifier(model, args)
    return model


def make_awesome_classifier(args, save=True):
    raise NotImplementedError('Not implemented yet')
    '''
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('[Training classifier...] Current cuda device: ', torch.cuda.current_device())

    model = ResNet50()
    model.to(device=device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # TODO: aug 넣기
    train_loader = create_classifier_data_loader(args.classifier_data_path_train, args, shuffle=True, isforward=False)
    val_loader = create_classifier_data_loader(args.classifier_data_path_val, args, shuffle=args.use_val_for_final, isforward=False)

    for epoch in range(args.classifier_epoch):
        train_loss = 0
        for iter, data in enumerate(train_loader):
            image_grappa, target = data
            image_grappa = image_grappa.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(image_grappa.unsqueeze(0))
            loss = loss_fn(output, target)
            l2_reg = torch.tensor(0., device=device)
            for p in model.parameters():
                l2_reg += torch.norm(p, 2)**2
            loss = loss + 2e-5 * l2_reg
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print(f'[Training classifier...] Epoch {epoch} | Training loss: {train_loss}')

        if args.use_val_for_final:
            train_loss = 0
            for iter, data in enumerate(val_loader):
                image_grappa, target = data
                image_grappa = image_grappa.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(image_grappa.unsqueeze(0))
                loss = loss_fn(output, target)
                l2_reg = torch.tensor(0., device=device)
                for p in model.parameters():
                    l2_reg += torch.norm(p, 2)**2
                loss = loss + 2e-5 * l2_reg
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(val_loader)
            print(f'[Training classifier for val set...] Epoch {epoch} | Training loss: {train_loss}')
        else:
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
    
    model.eval()

    if save:
        save_classifier(model, args)
    return model
    '''