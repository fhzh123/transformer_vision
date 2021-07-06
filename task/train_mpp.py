# Import modules
import os
import gc
import time
import logging
import json
# Import PyTorch
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.MPP.mpp import MPP
from model.MPP.vit import ViT
from optimizer.utils import shceduler_select, optimizer_select
from utils import label_smoothing_loss, TqdmLoggingHandler, write_log


def train_epoch(args, epoch, model, dataloader, optimizer, scheduler, scaler, logger, device):

    # Train setting
    start_time_e = time.time()
    model = model.train()

    for i, (img, label) in enumerate(dataloader):
        # Optimizer setting
        optimizer.zero_grad()

        # Input, output setting
        img = img.to(device, non_blocking=True)

        # Model
        with autocast():
            loss = model(img)

        # Back-propagation
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if args.scheduler in ['constant', 'warmup']:
            scheduler.step()
        if args.scheduler == 'reduce_train':
            scheduler.step(loss)

        # Print loss value only training
        if i == 0 or freq == args.print_freq or i==len(dataloader):
            batch_log = "[Epoch:%d][%d/%d] train_loss:%2.3f  | learning_rate:%3.6f | spend_time:%3.2fmin" \
                    % (epoch+1, i, len(dataloader), 
                    loss.item(), optimizer.param_groups[0]['lr'], 
                    (time.time() - start_time_e) / 60)
            write_log(logger, batch_log)
            freq = 0
        freq += 1

def valid_epoch(args, model, dataloader, device):

    # Validation setting
    model = model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for i, (img, label) in enumerate(dataloader):

            # Input, output setting
            img = img.to(device, non_blocking=True)


            # Model
            with autocast():
                loss = model(img)

            # Print loss value only training
            val_loss += loss.item()

    return val_loss


def mpp_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #===================================#
    #==============Logging==============#
    #===================================#

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    #===================================#
    #============Data Load==============#
    #===================================#

    # 2) Dataloader setting
    write_log(logger, "Load data...")
    gc.disable()
    transform_dict = {
        'train': transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        'valid': transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }

    dataset = torchvision.datasets.ImageFolder('../dataset/tiny-imagenet-200/tiny-imagenet-200/train', 
                                                transform=transform_dict['train'])
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [80000, 10000, 10000])


    dataset_dict = {
        'train': train_data,
        'valid': val_data
    }

    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
                            # num_workers=0),
        'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            num_workers=args.num_workers)
    }
    gc.enable()
    write_log(logger, f"Total number of trainingsets iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Model setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, "Instantiating models...")
    vit = ViT(
                image_size=256,
                patch_size=32,
                num_classes=1000,
                dim=1024,
                depth=6,
                heads=8,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
                )

    mpp_trainer = MPP(
                transformer=vit,
                patch_size=32,
                dim=1024,
                mask_prob=0.15,          # probability of using token in masked prediction task
                random_patch_prob=0.30,  # probability of randomly replacing a token being used for mpp
                replace_prob=0.50,       # probability of replacing a token being used for mpp with the mask token
            )
    #model = model.train()
    vit = vit.to(device)
    model = mpp_trainer.to(device)

    # 2) Optimizer setting
    optimizer = optimizer_select(model, args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)
    scaler = GradScaler()

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.captioning_save_path, 'checkpoint_cap.pth.tar'), map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model = model.train()
        model = model.to(device)
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_val_acc = 0
    best_val_loss = 1e+05

    
    write_log(logger, 'Train start!')

    for epoch in range(start_epoch, args.num_epochs):
        
        train_epoch(args, epoch, model, dataloader_dict['train'], optimizer, scheduler, scaler, logger, device)
        val_loss = valid_epoch(args, model, dataloader_dict['valid'], device)

        val_loss /= len(dataloader_dict['valid'])
        write_log(logger, 'Validation Loss: %3.3f' % val_loss)
        if val_loss < best_val_loss:
            write_log(logger, 'Checkpoint saving...')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }, os.path.join(args.mpp_save_path, f'checkpoint_cap.pth.tar'))
            best_val_loss = val_loss
            best_epoch = epoch
        else:
            else_log = f'Still {best_epoch} epoch loss({round(best_val_loss, 4)})% is better...'
            write_log(logger, else_log)

    # 3)
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Accuracy: {round(best_val_loss, 4)}')