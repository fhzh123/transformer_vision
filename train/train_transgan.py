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
from model.GAN.TransGAN import Discriminator, Generator# LinearLrDecay
from optimizer.utils import shceduler_select, optimizer_select
from utils import label_smoothing_loss, TqdmLoggingHandler, write_log
from torch.autograd import Variable
from functions import train, LinearLrDecay, train_epoch

            

def transgan_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#    if not os.path.exists(args.preprocess_path):
       #os.mkdir(args.preprocess_path)

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
        'train': CelebA(data_path=args.data_path,
                            transform=transform_dict['train']),
        #'valid': CelebA(data_path=args.data_path, 
                            #transform=transform_dict['valid'], phase='valid')
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.dis_batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers)
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

    # 2) Optimizer setting
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_model.parameters()),
                                        0.0001, (0, 0.99))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_model.parameters()),
                                        0.0001, (0, 0.99))

    gen_scheduler =  LinearLrDecay(gen_optimizer,  0.0001, 0.0, 0, 500000 * 5)
    dis_scheduler =  LinearLrDecay(gen_optimizer,  0.0001, 0.0, 0, 500000 * 5)
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

    global_steps = start_epoch * len(dataloader_dict['train'])

    for epoch in range(start_epoch, args.num_epochs):

        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        train_epoch(args, epoch,  global_steps, gen_model, dis_model, dataloader_dict['train'], gen_optimizer, dis_optimizer,lr_schedulers, device)
        
        global_steps +=1
