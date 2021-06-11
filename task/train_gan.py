# Import modules
from random import shuffle
from typing import Generator
from functions import copy_params
import os
import gc
import time
import logging
import numpy as np
from tqdm import tqdm
# Import PyTorch
import torch
import torch.nn as nn
import torchvision.datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid, save_image
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.GAN.dataset import CustomDataset
from model.GAN.TransGAN_Generator import Generator 
from model.GAN.TransGAN_Discriminator import Discriminator
from model.GAN.utils import compute_gradient_penalty, weights_init
from optimizer.utils import shceduler_select, optimizer_select
from utils import TqdmLoggingHandler, write_log  

import torchvision.datasets as datasets

def train_epoch(args,  epoch,  model_dict, dataloader, scaler_dict, optimizer_dict, scheduler_dict, logger, device):





    # Train setting
    start_time_e = time.time()


    for i, (img) in enumerate(tqdm(dataloader)):

    

        gen_avg_param = copy_params(model_dict['generator'])

     
        #
        # real_img = img.to(device, non_blocking=True)
       
        # Optimizer setting
        optimizer_dict['generator'].zero_grad()
        optimizer_dict['discriminator'].zero_grad()

        #===================================#
        #========Train Discriminator========#
        #===================================#
        
        # Input, output setting
        real_img = img.to(device, non_blocking=True)
        input_noise_dis = torch.randn(img.shape[0], args.latent_dim).to(device)


        with autocast():
        # Generate fake image
            fake_img = model_dict['generator'](input_noise_dis, epoch).detach()
            assert fake_img.size() == real_img.size()
            # Discriminate to real image
            real_validity = model_dict['discriminator'](real_img)
            fake_validity = model_dict['discriminator'](fake_img)
            
        # Discriminator loss(wgan-eps)
        gradient_penalty = compute_gradient_penalty(
            model_dict['discriminator'], real_img, phi=args.phi)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (args.phi ** 2)
        d_loss += (torch.mean(real_validity) ** 2) * 1e-3

        # Discriminator loss back-propagation   
        scaler_dict['discriminator'].scale(d_loss).backward()
        scaler_dict['discriminator'].unscale_(optimizer_dict['discriminator'])
        clip_grad_norm_(model_dict['discriminator'].parameters(), 5.)
        scaler_dict['discriminator'].step(optimizer_dict['discriminator'])
        scaler_dict['discriminator'].update()


        #===================================#
        #==========Train Generator==========#
        #===================================#

        input_noise_gen = torch.randn(args.gen_bs, args.latent_dim).to(device)

        with autocast():
            # Generate fake image
            fake_img = model_dict['generator'](input_noise_gen, epoch)
            fake_validity = model_dict['discriminator'](fake_img)
        
        # Generator loss
        g_loss = -torch.mean(fake_validity)

        # Generator loss back-propagation

        
        scaler_dict['generator'].scale(g_loss).backward()
        scaler_dict['generator'].unscale_(optimizer_dict['generator'])
        clip_grad_norm_(model_dict['generator'].parameters(), args.clip_grad_norm)
        scaler_dict['generator'].step(optimizer_dict['generator'])
        scaler_dict['generator'].update()


        if args.scheduler in ['constant', 'warmup', 'LinearLrDecay']:
            scheduler_dict['generator'].step()
            scheduler_dict['discriminator'].step()
        if args.scheduler == 'reduce_train':
            scheduler_dict['generator'].step(g_loss)
            scheduler_dict['discriminator'].step(d_loss)
        elif args.scheduler == 'None' :
             continue
        
        #moving average weight
        for p, avg_p in zip(model_dict['generator'].parameters(), gen_avg_param):
            avg_p.mul_(0.999).add(0.001, p.data) 

        # Print loss value onlty training
        if i == 0 or (i+1) % args.print_freq == 0 or (i+1)==len(dataloader):
            save_image(fake_img[:25], os.path.join(args.transgan_save_path, f'sampled_images.jpg'), 
                       nrow=5, normalize=True, scale_each=True)
            batch_log = "[Epoch:%d][%d/%d] g_loss:%2.3f | d_loss:%02.2f | learning_rate:%3.6f | spend_time:%3.2fmin" \
                    % (epoch+1, i+1, len(dataloader)+1, 
                    g_loss.item(), d_loss.item(), optimizer_dict['generator'].param_groups[0]['lr'], 
                    (time.time() - start_time_e) / 60)
            write_log(logger, batch_log)
        

# def valid_epoch(args, model_dict, dataloader, device):

#     # Validation setting
#     model_dict['generator'].eval()
#     model_dict['discriminator'].eval()
#     val_loss = 0
#     val_acc = 0
#     with torch.no_grad():
#         for i, img in enumerate(dataloader):

#             # Input, output setting
#             real_img = img.to(device, non_blocking=True)
#             input_noise_dis = torch.randn(img.size(0), args.latent_dim)
#             input_noise_gen = torch.randn(img.size(0), args.latent_dim)

#             # Model
#             logit = model_dict['discriminator'](real_img)
#             first_token = logit[:,0,:]

#             # Loss calculate
#             loss = F.cross_entropy(first_token, label)

#             # Print loss value only training
#             acc = (((first_token.argmax(dim=1) == label).sum()) / label.size(0)) * 100
#             val_loss += loss.item()
#             val_acc += acc.item()

#     return val_loss, val_acc

def transgan_training(args):
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
    transform_dict = {
        'train': transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),            
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
    dataset_dict = {
        'train': CustomDataset(data_path=args.transgan_data_path,
                            transform=transform_dict['train']),
        'valid': CustomDataset(data_path=args.transgan_data_path,
                            transform=transform_dict['valid'])
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], 
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        'valid': DataLoader(dataset_dict['valid'], 
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

    model_dict = {
        'generator': Generator(d_model=args.gf_dim, initial_depth = args.initial_depth,
                               n_head=args.n_head, bottom_width=args.bottom_width,
                                dropout=args.dropout),
        'discriminator': Discriminator(n_classes=1, d_model=args.df_dim, 
                                       n_head=args.n_head, 
                                       num_encoder_layer=args.num_encoder_layer, img_size=args.img_size, 
                                       patch_size=args.patch_size, dropout=args.dropout,
                                       diff_aug = args.diff_aug)
    }
    
    model_dict['generator'] = model_dict['generator'].train()
    model_dict['generator'] = model_dict['generator'].to(device)

    model_dict['discriminator'] = model_dict['discriminator'].train()
    model_dict['discriminator'] = model_dict['discriminator'].to(device)

    # 2) Optimizer setting
    optimizer_dict = {
        'generator': optimizer_select(model_dict['generator'], args),
        'discriminator': optimizer_select(model_dict['discriminator'], args)
    }
    scheduler_dict = {
        'generator': shceduler_select(optimizer_dict['generator'], dataloader_dict, args),
        'discriminator': shceduler_select(optimizer_dict['discriminator'], dataloader_dict, args)
    }
    scaler_dict = {
        'generator': GradScaler(),
        'discriminator': GradScaler()
    }

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.transgan_checkpt_save_path, 'checkpoint_gan.pth.tar'), map_location='cpu')
        # Model load
        model_dict['generator'].load_state_dict(checkpoint['gen_model'])
        model_dict['discriminator'].load_state_dict(checkpoint['dis_model'])
        optimizer_dict['generator'].load_state_dict(checkpoint['gen_optimizer'])
        optimizer_dict['discriminator'].load_state_dict(checkpoint['dis_optimizer'])

        if scheduler_dict['generator'] and scheduler_dict['discriminator'] is not None:
            scheduler_dict['generator'].load_state_dict(checkpoint['gen_scheduler'])
            scheduler_dict['discriminator'].load_state_dict(checkpoint['dis_scheduler'])
        
        scaler_dict['generator'].load_state_dict(checkpoint['gen_scaler'])
        scaler_dict['discriminator'].load_state_dict(checkpoint['dis_scaler'])
        # setting
        start_epoch = checkpoint['epoch'] + 1
        model_dict['generator'] = model_dict['generator'].train()
        model_dict['generator'] = model_dict['generator'].to(device)
        model_dict['discriminator'] = model_dict['discriminator'].train()
        model_dict['discriminator'] = model_dict['discriminator'].to(device)
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'Train start!')
    saving = True

    #weight initialize
    model_dict['generator'].apply(weights_init)
    model_dict['discriminator'].apply(weights_init)
    

    for epoch in range(start_epoch, args.num_epochs):



        train_epoch(args, epoch,  model_dict, dataloader_dict['train'], scaler_dict,
                    optimizer_dict,  logger, device)
        # val_loss, val_acc = valid_epoch(args, model_dict, dataloader_dict['valid'], device)

        # val_loss /= len(dataloader_dict['valid'])
        # val_acc /= len(dataloader_dict['valid'])
        # write_log(logger, 'Validation Loss: %3.3f' % val_loss)
        # write_log(logger, 'Validation Accuracy: %3.2f%%' % val_acc)
        if saving:
            write_log(logger, 'Checkpoint saving...')
            torch.save({
                'epoch': epoch,
                'gen_model': model_dict['generator'].state_dict(),
                'dis_model': model_dict['discriminator'].state_dict(),
                'gen_optimizer': optimizer_dict['generator'].state_dict(),
                'dis_optimizer': optimizer_dict['discriminator'].state_dict(),
                'gen_scaler': scaler_dict['generator'].state_dict(),
                'dis_scaler': scaler_dict['discriminator'].state_dict()
            }, os.path.join(args.transgan_checkpt_save_path, f'checkpoint_gan.pth.tar'))
            best_epoch = epoch
        # else:
        #     else_log = f'Still {best_epoch} epoch accuracy({round(best_val_acc, 2)})% is better...'
        #     write_log(logger, else_log)