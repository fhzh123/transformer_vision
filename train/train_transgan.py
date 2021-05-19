# Import modules
from model.GAN.CelebA import CelebA
import os
import gc
import time
import logging
import numpy as np
from tqdm import tqdm
# Import PyTorch
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.classification.dataset import CustomDataset
from model.GAN.TransGAN import Discriminator, Generator, LinearLrDecay
from optimizer.utils import shceduler_select, optimizer_select
from utils import label_smoothing_loss, TqdmLoggingHandler, write_log


def get_attn_mask(N, w):
    mask = torch.zeros(1, 1, N, N).cuda()
    for i in range(N):
        if i <= w:
            mask[:, :, i, 0:i+w+1] = 1
        elif N - i <= w:
            mask[:, :, i, i-w:N] = 1
        else:
            mask[:, :, i, i:i+w+1] = 1
            mask[:, :, i, i-w:i] = 1
    return mask




def train_epoch(args, epoch, gen_net: nn.Module, dis_net: nn.Module, dataloader, gen_optimizer, dis_optimizer, gen_scheduler, dis_scheduler, scaler, device):

    # Train setting
    start_time_e = time.time()
    gen_model = gen_net.train()
    dis_model = dis_net.train()
    #tgt_mask = gen_model.generate_square_subsequent_mask(args.max_len - 1, device)

    gen_step = 0

   

    #scheduler 


    for i, img in enumerate(dataloader):

     
         # Optimizer setting
        gen_optimizer.zero_grad()
        dis_optimizer.zero_grad()

        # Adversarial ground truths

        real_img = img.to(device, non_blocking=True)

        #Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (img.shape[0], args.latent_dim) )).to(device)

        #Train Discriminator

        real_validity = dis_net(real_img)
        fake_img = gen_net(z)
        assert fake_img.size() == real_img.size(), f"fake_imgs.size(): {fake_img.size()} real_imgs.size(): {real_img.size()}"
        fake_validity = dis_net(fake_img)

        #Discriminator loss
        d_loss = 0
        for real_validity_item, fake_validity_item in zip(real_validity, fake_validity):
            real_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 1., dtype=torch.float, device=real_img.get_device())
            fake_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 0., dtype=torch.float, device=real_img.get_device())
            d_real_loss = nn.MSELoss()(real_validity_item, real_label)
            d_fake_loss = nn.MSELoss()(fake_validity_item, fake_label)
            d_loss += d_real_loss + d_fake_loss

        #Train Generator
        gen_optimizer.zero_grad()

        gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (img.shape[0], args.latent_dim) )).to(device)
        gen_imgs = gen_net(gen_z)
        fake_validity = dis_net(gen_imgs)
        
        g_loss = 0
        #if isinstance(fake_validity, list): #need to check

        for fake_validity_item in fake_validity:
            real_label = torch.full((fake_validity_item.shape[0],fake_validity_item.shape[1]), 1., dtype=torch.float, device=real_img.get_device())
            g_loss += nn.MSELoss()(fake_validity_item, real_label)

        # Back-propagation
        scaler.scale(d_loss).backward()
        scaler.scale(g_loss).backward()
        scaler.unscale_(gen_optimizer)
        scaler.unscale_(dis_optimizer)
        clip_grad_norm_(gen_model.parameters(), args.clip_grad_norm)
        clip_grad_norm_(dis_model.parameters(), args.clip_grad_norm)
        scaler.step(gen_optimizer)
        scaler.step(dis_optimizer)
        scaler.update()


        # Print loss value only training
        if gen_step and i % args.print_freq == 0:
            sample_imgs = gen_imgs[:25]
            # scale_factor = args.img_size // int(sample_imgs.size(3))
            # sample_imgs = torch.nn.functional.interpolate(sample_imgs, scale_factor=2)
            img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
            save_image(sample_imgs, f'sampled_images_{args.exp_name}.jpg', nrow=5, normalize=True, scale_each=True)
            # writer.add_image(f'sampled_images_{args.exp_name}', img_grid, global_steps)
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, i % len(dataloader), len(dataloader), d_loss.item(), g_loss.item()))


def valid_epoch(args, model, dataloader, device):

    # Validation setting
    model = model.eval()
    val_loss = 0
    val_acc = 0


    with torch.no_grad():
        for i, (img, caption) in enumerate(dataloader):

            # Input, output setting
            img = img.to(device, non_blocking=True)
            caption = caption.long().to(device, non_blocking=True)

            label = caption[:, 1:]
            non_pad = label != args.pad_id
            label = label[non_pad].contiguous().view(-1)

            # Model
            with autocast():
                predicted = model(
                    img, caption[:, :-1], tgt_mask, non_pad_position=non_pad)
                predicted = predicted.view(-1, predicted.size(-1))
                loss = F.cross_entropy(
                    predicted, label, ignore_index=args.pad_id)

            # Print loss value only training
            acc = (predicted.argmax(dim=1) == label).sum() / len(label)
            val_loss += loss.item()
            val_acc += (acc.item() * 100)

    return val_loss, val_acc

def transgan_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.preprocess_path):
        os.mkdir(args.preprocess_path)

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
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ColorJitter(brightness=(0.5, 2)),
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
        'train': CelebA(data_path=args.data_path,
                            transform=transform_dict['train']),
        #'valid': CelebA(data_path=args.data_path, 
                            #transform=transform_dict['valid'], phase='valid')
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
        #'valid': DataLoader(dataset_dict['valid'], drop_last=False,
                            #batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            #num_workers=args.num_workers)
    }
    gc.enable()
    write_log(logger, f"Total number of trainingsets iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")

    #===================================#
    #===========Model setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, "Instantiating models...")
    gen_model = Generator(args)
    dis_model = Discriminator(args)
    
    gen_model = gen_model.train()
    gen_model = gen_model.to(device)

    dis_model = dis_model.train()
    dis_model = dis_model.to(device)

    # 2) Optimizer setting
    gen_optimizer = optimizer_select(gen_model, args)
    dis_optimizer = optimizer_select(dis_model, args)
    #gen_scheduler = shceduler_select(gen_optimizer, dataloader_dict, args)
    #dis_scheduler = shceduler_select(dis_optimizer, dataloader_dict, args)
    gen_scheduler =  LinearLrDecay(gen_optimizer,  0.0001, 0.0, 0, 500000 * 5)
    dis_scheduler =  LinearLrDecay(gen_optimizer,  0.0001, 0.0, 0, 500000 * 5)
    scaler = GradScaler()

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.model_path, 'gan_checkpoint.pth.tar'), map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        gen_model.load_state_dict(checkpoint['gen_model'])
        dis_model.load_state_dict(checkpoint['dis_model'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        gen_scheduler.load_state_dict(checkpoint['gen_scheduler'])
        dis_scheduler.load_state_dict(checkpoint['dis_scheduler'])
        gen_model = gen_model.train()
        gen_model = gen_model.to(device)
        dis_model = dis_model.train()
        dis_model = dis_model.to(device)
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    best_val_acc = 0

    write_log(logger, 'Train start!')

    for epoch in range(start_epoch, args.num_epochs):

        train_epoch(args, epoch,  gen_model, dis_model, dataloader_dict['train'], gen_optimizer, dis_optimizer, gen_scheduler, dis_scheduler, scaler, device)
        #val_loss, val_acc = valid_epoch(args, model, dataloader_dict['valid'], device)

        #val_loss /= len(dataloader_dict['valid'])
        #val_acc /= len(dataloader_dict['valid'])
        #write_log(logger, 'Validation Loss: %3.3f' % val_loss)
        #write_log(logger, 'Validation Accuracy: %3.2f%%' % val_acc)
        #if val_acc > best_val_acc:
            #write_log(logger, 'Checkpoint saving...')
            #torch.save({
                #'epoch': epoch,
                #'model': model.state_dict(),
                #'optimizer': optimizer.state_dict(),
                #'scheduler': scheduler.state_dict(),
                #'scaler': scaler.state_dict()
            #}, #f'checkpoint.pth.tar')
            #best_val_acc = val_acc
            #best_epoch = epoch
        #else:
            #else_log = f'Still {best_epoch} epoch accuracy({round(best_val_acc, 2)})% is better...'
            #write_log(logger, else_log)

    # 3)
    #print(f'Best Epoch: {best_epoch}')
    #print(f'Best Accuracy: {round(best_val_acc, 2)}')