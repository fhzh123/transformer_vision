import logging
import operator
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from imageio import imsave
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import cv2
from torch.nn.utils import clip_grad_norm_
import time

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

def train_epoch(args, epoch, global_steps, gen_net: nn.Module, dis_net: nn.Module, dataloader,   gen_optimizer, dis_optimizer, device, schedulers =None):
    gen_step = 0
    # Train setting
    start_time_e = time.time()
    gen_model = gen_net.train()
    dis_model = dis_net.train()
    dis_model.module.cur_stage = gen_model.module.cur_stage
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
        if args.loss == 'hinge':
            d_loss = 0
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                    torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        elif args.loss == 'standard':
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
        if args.loss == "lsgan":
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
                #sample_imgs = [gen_imgs_list[i] for i in range(len(gen_imgs_list))]
        # scale_factor = args.img_size // int(sample_imgs.size(3))
        # sample_imgs = torch.nn.functional.interpolate(sample_imgs, scale_factor=2)
        #img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
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
            'dis_optimizer': dis_optimizer.state_dict(),
            'scaler': scaler.state_dict()
        }, os.path.join(args.save_path, 'gan_checkpoint.pth.tar'))

        # writer.add_image(f'sa
def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
          epoch, writer_dict, fixed_z, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    dis_net.module.cur_stage = gen_net.module.cur_stage
    

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']
        if gen_net.module.alpha < 1:
            gen_net.module.alpha += args.fade_in
            gen_net.module.alpha = min(1., gen_net.module.alpha)
            dis_net.module.alpha = gen_net.module.alpha

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor).to("cuda:0")

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))).to(real_imgs.get_device())

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z, epoch).detach()
        assert fake_imgs.size() == real_imgs.size(), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"

        fake_validity = dis_net(fake_imgs)

        # cal loss
        if args.loss == 'hinge':
            d_loss = 0
            d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                    torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        elif args.loss == 'standard':
            real_label = torch.full((imgs.shape[0],), 1., dtype=torch.float, device=real_imgs.get_device())
            fake_label = torch.full((imgs.shape[0],), 0., dtype=torch.float, device=real_imgs.get_device())
            real_validity = nn.Sigmoid()(real_validity.view(-1))
            fake_validity = nn.Sigmoid()(fake_validity.view(-1))
            d_real_loss = nn.BCELoss()(real_validity, real_label)
            d_fake_loss = nn.BCELoss()(fake_validity, fake_label)
        elif args.loss == 'lsgan':
            if isinstance(fake_validity, list):
                d_loss = 0
                for real_validity_item, fake_validity_item in zip(real_validity, fake_validity):
                    real_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                    fake_label = torch.full((real_validity_item.shape[0],real_validity_item.shape[1]), 0., dtype=torch.float, device=real_imgs.get_device())
                    d_real_loss = nn.MSELoss()(real_validity_item, real_label)
                    d_fake_loss = nn.MSELoss()(fake_validity_item, fake_label)
                    d_loss += d_real_loss + d_fake_loss
            else:
                real_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                fake_label = torch.full((real_validity.shape[0],real_validity.shape[1]), 0., dtype=torch.float, device=real_imgs.get_device())
                d_real_loss = nn.MSELoss()(real_validity, real_label)
                d_fake_loss = nn.MSELoss()(fake_validity, fake_label)
                d_loss = d_real_loss + d_fake_loss
        elif args.loss == 'wgangp-eps':
            gradient_penalty = compute_gradient_penalty(dis_net, real_imgs, fake_imgs.detach(), args.phi)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10 / (
                    args.phi ** 2)
            d_loss += (torch.mean(real_validity) ** 2) * 1e-3
        else:
            raise NotImplementedError(args.loss)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()

            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z, epoch)
            fake_validity = dis_net(gen_imgs)

            # cal loss
            if args.loss == "standard":
                real_label = torch.full((args.gen_batch_size,), 1., dtype=torch.float, device=real_imgs.get_device())
                fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                g_loss = nn.BCELoss()(fake_validity.view(-1), real_label)
            if args.loss == "lsgan":
                if isinstance(fake_validity, list):
                    g_loss = 0
                    for fake_validity_item in fake_validity:
                        real_label = torch.full((fake_validity_item.shape[0],fake_validity_item.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                        g_loss += nn.MSELoss()(fake_validity_item, real_label)
                else:
                    real_label = torch.full((fake_validity.shape[0],fake_validity.shape[1]), 1., dtype=torch.float, device=real_imgs.get_device())
                    # fake_validity = nn.Sigmoid()(fake_validity.view(-1))
                    g_loss = nn.MSELoss()(fake_validity, real_label)
            else:
                g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            sample_imgs = gen_imgs[:25]
            # scale_factor = args.img_size // int(sample_imgs.size(3))
            # sample_imgs = torch.nn.functional.interpolate(sample_imgs, scale_factor=2)
            img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
            save_image(sample_imgs, f'sampled_images_{args.exp_name}.jpg', nrow=5, normalize=True, scale_each=True)
            # writer.add_image(f'sampled_images_{args.exp_name}', img_grid, global_steps)
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] alpha: %f" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item(), gen_net.module.alpha))

        writer_dict['train_global_steps'] = global_steps + 1

class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten