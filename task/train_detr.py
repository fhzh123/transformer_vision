import gc
import os
import time
import logging

import torch
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log, label_smoothing_loss

from model.detection.datasets import build_dataset
from model.detection.datasets.coco import make_coco_transforms, build
from model.detection.util import misc as utils
from model.detection import models 


def train_epoch(args, epoch, model, criterion, dataloader, optimizer, scheduler, scaler, logger, device):

    # Train setting
    start_time_e = time.time()
    model = model.train()

    for i, (img, targets) in enumerate(dataloader):

        optimizer.zero_grad()

        img = img.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(img)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        if args.scheduler in ['constant', 'warmup']:
            scheduler.step()
        # if args.scheduler == 'reduce_train':
        #     scheduler.step(mlm_loss)

        if i == 0 or freq == args.print_freq or i==len(dataloader):
            batch_log = "[Epoch:%d][%d/%d] train_loss:%2.3f  | learning_rate:%3.6f | spend_time:%3.2fmin" \
                    % (epoch+1, i, len(dataloader), 
                    loss.item(), optimizer.param_groups[0]['lr'], 
                    (time.time() - start_time_e) / 60)
            write_log(logger, batch_log)
            freq = 0
        freq += 1




def valid_epoch(args, model, criterion, dataloader, device):
    model.eval()
    criterion.eval()

    with torch.no_grad():
        for i, (img, targets) in enumerate(dataloader):
            img = img.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(img)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

    return loss_dict


def detr_training(args):
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

    # 1) Dataloader setting
    write_log(logger, "Load data...")
    gc.disable()    

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    transform_dict = {
        'train' : make_coco_transforms('train'),
        'valid' : make_coco_transforms('val')
    }

    dataset_dict = {
        'train' : build('train', args),
        'valid' : build('val', args)
    }

    dataloader_dict = {
        'train' : DataLoader(dataset_dict['train'], batch_sampler=batch_sampler_train, pin_memory=True, 
                            collate_fn=utils.collate_fn, num_workers=args.num_workers),
        'valid' : DataLoader(dataset_dict['valid'], sampler=sampler_val, pin_memory=True, 
                            collate_fn=utils.collate_fn, num_workers=args.num_workers) 
    }


    gc.enable()
    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Model setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, "Instantiating models...")

    model, criterion, postprocessors = models.build_model(args)
    model = model.train()
    model = model.to(device)

    optimizer = optimizer_select(model, args)
    scheduler = shceduler_select(optimizer, dataloader_dict, args)
    scaler = GradScaler()

    # 2) Model resume

    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(os.path.join(args.model_path, 'checkpoint.pth.tar'), map_location='cpu')
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

    best_val_loss = 1e05

    write_log(logger, 'Train start!')

    for epoch in range(start_epoch, args.num_epochs):
        train_epoch(args, epoch, model, criterion, dataloader_dict['train'], optimizer, scheduler, scaler, logger, device)

        loss_dict = valid_epoch(args, model, criterion, dataloader_dict['valid'], device)
        weight_dict = criterion.weight_dict
        val_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        val_loss /= len(dataloader_dict['valid'])
        write_log(logger, 'Validation Loss: %3.3f' % val_loss)

        if val_loss < best_val_loss:
            write_log(logger, 'Checkpoint saving...')

            if not os.path.exists(args.detr_save_path):
                os.mkdir(args.detr_save_path)

            torch.save({
                'epoch' : epoch,
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'scaler' : scaler.state_dict()
            }, os.path.join(args.detr_save_path, f'checkpoint.pth.tar'))
            best_val_loss = val_loss
            best_epoch = epoch

        else:
            else_log = f'Still {best_epoch} epoch val loss {best_val_loss} if better...'
            write_log(logger, else_log)

    write_log(logger, f'Best Epoch : {best_epoch}')
    write_log(logger, f'Best Loss : {best_val_loss}')


