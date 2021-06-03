# Import modules
import os
import gc
import time
import logging
import sentencepiece as spm
# Import PyTorch
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.captioning.dataset import CustomDataset
from model.captioning.captioning_model import Vision_Transformer
from optimizer.utils import shceduler_select, optimizer_select
from utils import label_smoothing_loss, TqdmLoggingHandler, write_log

def preprocessing(args):
    with open(os.path.join(args.data_path, 'annotations/captions_train2017.json'), 'r') as f:
        data_ = json.load(f)['annotations']

    #===================================#
    #===========SentencePiece===========#
    #===================================#

    # 1) Make Korean text to train vocab
    with open(f'{args.preprocess_path}/train_text.txt', 'w') as f:
        for c in data_:
            f.write(f'{c["caption"]}\n')

    # 2) SentencePiece model training
    spm.SentencePieceProcessor()
    spm.SentencePieceTrainer.Train(
        f'--input={args.preprocess_path}/train_text.txt --model_prefix={args.preprocess_path}/spm_train_{args.vocab_size} '
        f'--model_type={args.sentencepiece_model} --character_coverage=0.9995 --vocab_size={args.vocab_size} '
        f'--pad_id={args.pad_id} --unk_id={args.unk_id} --bos_id={args.bos_id} --eos_id={args.eos_id} '
        f'--split_by_whitespace=true')

def train_epoch(args, epoch, model, dataloader, optimizer, scheduler, scaler, logger, device):

    # Train setting
    start_time_e = time.time()
    model = model.train()
    tgt_mask = model.generate_square_subsequent_mask(args.max_len - 1, device)

    for i, (img, caption) in enumerate(dataloader):

        # Optimizer setting
        optimizer.zero_grad()

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
            loss = label_smoothing_loss(
                predicted, label, device)

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
        acc = (predicted.argmax(dim=1) == label).sum() / len(label)
        if i == 0 or freq == args.print_freq or i==len(dataloader):
            batch_log = "[Epoch:%d][%d/%d] train_loss:%2.3f  | train_acc:%02.2f | learning_rate:%3.6f | spend_time:%3.2fmin" \
                    % (epoch+1, i, len(dataloader), 
                    loss.item(), acc.item() * 100, optimizer.param_groups[0]['lr'], 
                    (time.time() - start_time_e) / 60)
            write_log(logger, batch_log)
            freq = 0
        freq += 1

def valid_epoch(args, model, dataloader, device):

    # Validation setting
    model = model.eval()
    val_loss = 0
    val_acc = 0
    tgt_mask = model.generate_square_subsequent_mask(args.max_len - 1, device)

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

def captioning_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.captioning_preprocess_path):
        os.mkdir(args.captioning_preprocess_path)

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

    # 1) SentencePiece load
    if not os.path.isfile(os.path.join(args.captioning_preprocess_path, f'spm_train_{args.vocab_size}.model')):
        preprocessing(args)

    spm_model = spm.SentencePieceProcessor()
    spm_model.Load(os.path.join(args.captioning_preprocess_path, f'spm_train_{args.vocab_size}.model'))

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
    dataset_dict = {
        'train': CustomDataset(data_path=args.captioning_data_path, spm_model=spm_model,
                            transform=transform_dict['train'], phase='train',
                            min_len=args.min_len, max_len=args.max_len),
        'valid': CustomDataset(data_path=args.captioning_data_path, spm_model=spm_model,
                            transform=transform_dict['valid'], phase='valid',
                            min_len=args.min_len, max_len=args.max_len)
    }
    dataloader_dict = {
        'train': DataLoader(dataset_dict['train'], drop_last=True,
                            batch_size=args.batch_size, shuffle=True, pin_memory=True,
                            num_workers=args.num_workers),
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
    model = Vision_Transformer(trg_vocab_num=args.vocab_size, d_model=args.d_model, d_embedding=args.d_embedding, 
                               n_head=args.n_head, dim_feedforward=args.dim_feedforward, img_size=args.img_size, 
                               patch_size=args.patch_size, max_len=args.max_len, pad_id=args.pad_id, 
                               num_encoder_layer=args.num_encoder_layer, num_decoder_layer=args.num_decoder_layer,
                               dropout=args.dropout, embedding_dropout=args.embedding_dropout, parallel=args.parallel,
                               triple_patch=args.triple_patch)
    model = model.train()
    model = model.to(device)

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

    write_log(logger, 'Train start!')

    for epoch in range(start_epoch, args.num_epochs):

        train_epoch(args, epoch, model, dataloader_dict['train'], optimizer, scheduler, scaler, logger, device)
        val_loss, val_acc = valid_epoch(args, model, dataloader_dict['valid'], device)

        val_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Validation Loss: %3.3f' % val_loss)
        write_log(logger, 'Validation Accuracy: %3.2f%%' % val_acc)
        if val_acc > best_val_acc:
            write_log(logger, 'Checkpoint saving...')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }, os.path.join(args.captioning_save_path, f'checkpoint_cap.pth.tar'))
            best_val_acc = val_acc
            best_epoch = epoch
        else:
            else_log = f'Still {best_epoch} epoch accuracy({round(best_val_acc, 2)})% is better...'
            write_log(logger, else_log)

    # 3)
    print(f'Best Epoch: {best_epoch}')
    print(f'Best Accuracy: {round(best_val_acc, 2)}')