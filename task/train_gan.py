# Import modules
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
from models.GAN.dataset import CustomDataset
from model.GAN.TransGAN import Discriminator, Generator# LinearLrDecay
from optimizer.utils import shceduler_select, optimizer_select
from utils import label_smoothing_loss, TqdmLoggingHandler, write_log
from torch.autograd import Variable
from copy import deepcopy

def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty

def train_epoch(args, epoch, gen_net: nn.Module, dis_net: nn.Module, dataloader,   gen_optimizer, dis_optimizer, device, schedulers =None):
    gen_step = 0
    # Train setting
    start_time_e = time.time()
    gen_model = gen_net.train()
    dis_model = dis_net.train()
    #tgt_mask = gen_model.generate_square_subsequent_mask(args.max_len - 1, device)
    #scheduler 
    for i, img in enumerate(tqdm(dataloader)):

         # Optimizer setting
        #real_img = img.to(device, non_blocking=True)
        real_img = img.type(torch.cuda.FloatTensor).to("cuda:0")

        #Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (img.shape[0], 1024)))
        dis_optimizer.zero_grad()

        #Train Discriminator
        real_validity = dis_net(real_img)
        fake_img = gen_net(z, epoch).detach()
        assert fake_img.size() == real_img.size(), f"fake_img.size(): {fake_img.size()} real_img.size(): {real_img.size()}"
        fake_validity = dis_net(fake_img)

        #Discriminator loss
        # cal loss
        if args.loss == 'standard':
            real_label = torch.full((img.shape[0],), 1., dtype=torch.float, device=real_img.get_device())
            fake_label = torch.full((img.shape[0],), 0., dtype=torch.float, device=real_img.get_device())
            real_validity = nn.Sigmoid()(real_validity.view(-1))
            fake_validity = nn.Sigmoid()(fake_validity.view(-1))
            d_real_loss = nn.BCELoss()(real_validity, real_label)
            d_fake_loss = nn.BCELoss()(fake_validity, fake_label)

        elif args.loss == 'lsgan':
            if isinstance(fake_validity, list):
                d_loss = 0
                for real_validity_item, fake_validity_item in zip(real_validity, fake_validity):
                    real_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 1., dtype=torch.float, device=real_img.get_device())
                    fake_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 0., dtype=torch.float, device=real_img.get_device())
                    d_real_loss = nn.MSELoss()(real_validity_item, real_label)
                    d_fake_loss = nn.MSELoss()(fake_validity_item, fake_label)
                    d_loss += d_real_loss + d_fake_loss
            else:
                real_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 1., dtype=torch.float, device=real_img.get_device())
                fake_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 0., dtype=torch.float, device=real_img.get_device())
                d_real_loss = nn.MSELoss()(real_validity, real_label)
                d_fake_loss = nn.MSELoss()(fake_validity, fake_label)
                d_loss = d_real_loss + d_fake_loss

        elif args.loss == 'wgangp-eps':
            gradient_penalty = compute_gradient_penalty(dis_net, real_img, fake_img.detach(), phi=1)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    1 ** 2)
            d_loss += (torch.mean(real_validity) ** 2) * 1e-3
        else:
            raise NotImplementedError(args.loss)

        d_loss.backward()
        clip_grad_norm_(dis_model.parameters(), args.clip_grad_norm)
        dis_optimizer.step()

        #Train Generator
        gen_optimizer.zero_grad()
        gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, 1024) ))
        gen_imgs = gen_net(gen_z, epoch)
        fake_validity = dis_net(gen_imgs)
        
        if args.loss == "standard":
            real_label = torch.full((args.gen_batch_size,), 1., dtype=torch.float, device=real_img.get_device())
            fake_validity = nn.Sigmoid()(fake_validity.view(-1))
            g_loss = nn.BCELoss()(fake_validity.view(-1), real_label)

        elif args.loss == "lsgan":
            if isinstance(fake_validity, list):
                g_loss = 0
                for fake_validity_item in fake_validity:
                    real_label = torch.full((fake_validity_item.shape[0],fake_validity_item.shape[1]), 1., dtype=torch.float, device=real_img.get_device())
                    g_loss += nn.MSELoss()(fake_validity_item, real_label)
            else:
                real_label = torch.full((fake_validity.shape[0],fake_validity.shape[1]), 1., dtype=torch.float, device=real_img.get_device())
                # fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                g_loss = nn.MSELoss()(fake_validity, real_label)
        else:
            g_loss = -torch.mean(fake_validity)

        g_loss.backward()
        clip_grad_norm_(gen_model.parameters(), 5.)
        gen_optimizer.step()
        # Back-propagation
        
        gen_step += 1

        with torch.no_grad():
            # Print loss value only training
            if gen_step and i % args.print_freq == 0:
                sample_imgs=gen_imgs[:16]
                save_image(sample_imgs, f'sampled_images_{args.exp_name}.jpg', nrow=5, normalize=True, scale_each=True)
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                     (epoch, args.num_epochs, i % len(dataloader), len(dataloader), d_loss.item(), g_loss.item()))

    if epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'gen_model': gen_model.state_dict(),
            'dis_model': dis_model.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict()
        }, os.path.join(args.save_path, 'gan_checkpoint.pth.tar'))


def transgan_training(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    }
    dataset_dict = {
        'train': CustomDataset(data_path=args.data_path,
                            transform=transform_dict['train'])
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.dis_batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)
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

    gen_model = gen_model.to(device)
    dis_model = dis_model.to(device)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_model.apply(weights_init)
    dis_model.apply(weights_init)


    gpu_ids = [i for i in range(int(torch.cuda.device_count()))]
    gen_model = torch.nn.DataParallel(gen_model)
    dis_model = torch.nn.DataParallel(dis_model)

    gen_model.module.cur_stage = 0
    dis_model.module.cur_stage = 0
    gen_model.module.alpha = 1.
    dis_model.module.alpha = 1.

    # 2) Optimizer setting
    gen_optimizer = optimizer_select(gen_model, args)
    dis_optimizer = optimizer_select(dis_model, args)

    gen_scheduler = shceduler_select(gen_optimizer, dataloader_dict, args)
    dis_scheduler = shceduler_select(dis_optimizer, dataloader_dict, args)
    #scaler = GradScaler()

    # 3) Model resume
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.save_path, 'gan_checkpoint.pth.tar'), map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        gen_model.load_state_dict(checkpoint['gen_model'])
        dis_model.load_state_dict(checkpoint['dis_model'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        gen_model = gen_model.train()
        gen_model = gen_model.to(device)
        dis_model = dis_model.train()
        dis_model = dis_model.to(device)
        del checkpoint

    #===================================#
    #=========Model Train Start=========#
    #===================================#

    write_log(logger, 'Train start!')

    for epoch in range(start_epoch, args.num_epochs):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        train_epoch(args, epoch, gen_model, dis_model, dataloader_dict['train'], gen_optimizer, dis_optimizer,lr_schedulers, device)
