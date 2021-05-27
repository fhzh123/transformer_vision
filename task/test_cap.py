# Import modules
import os
import gc
import time
import logging
import sentencepiece as spm
import matplotlib.pyplot as plt
from math import ceil, sqrt
from collections import defaultdict
# Import PyTorch
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.captioning.dataset import CustomDataset
from model.captioning.captioning_model import Vision_Transformer
from utils import TqdmLoggingHandler, write_log, UnNormalize

def captioning_testing(args):
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

    # 1) SentencePiece load
    spm_model = spm.SentencePieceProcessor()
    spm_model.Load(os.path.join(args.captioning_preprocess_path, f'spm_train_{args.vocab_size}.model'))

    # 2) Dataloader setting
    write_log(logger, "Load data...")
    gc.disable()
    transform_test = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    dataset_test = CustomDataset(data_path=args.captioning_data_path, spm_model=spm_model,
                                 transform=transform_test, phase='valid',
                                 min_len=args.min_len, max_len=args.max_len)
    dataloader_test = DataLoader(dataset_test, drop_last=False,
                                 batch_size=args.test_batch_size, shuffle=True, pin_memory=True,
                                 num_workers=args.num_workers)
    gc.enable()
    write_log(logger, f"Total number of testingsets iterations - {len(dataset_test)}, {len(dataloader_test)}")

    #===================================#
    #===========Model setting===========#
    #===================================#

    # 1) Model initiating
    write_log(logger, "Loading models...")
    model = Vision_Transformer(trg_vocab_num=args.vocab_size, d_model=args.d_model, d_embedding=args.d_embedding, 
                               n_head=args.n_head, dim_feedforward=args.dim_feedforward, img_size=args.img_size, 
                               patch_size=args.patch_size, max_len=args.max_len, pad_id=args.pad_id, 
                               num_encoder_layer=args.num_encoder_layer, num_decoder_layer=args.num_decoder_layer,
                               dropout=args.dropout, embedding_dropout=args.embedding_dropout, parallel=args.parallel,
                               triple_patch=args.triple_patch, device=device)

    # 2) Model load
    checkpoint = torch.load(os.path.join(args.captioning_save_path, 'checkpoint_cap.pth.tar'))
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    model = model.to(device)
    del checkpoint

    #===================================#
    #=========Beam Search Start=========#
    #===================================#

    predicted_list = list()
    label_list = list()
    every_batch = torch.arange(0, args.beam_size * args.test_batch_size, args.beam_size, device=device)
    tgt_masks = {l: model.generate_square_subsequent_mask(l, device) for l in range(1, args.max_len + 1)}
    start_time = time.time()

    write_log(logger, "Test start!")
    with torch.no_grad():
        for print_step, (img, caption) in enumerate(dataloader_test):

            img = img.to(device)
            caption_list = caption.tolist()
            label_list.extend(caption_list)
            encoder_out_dict = defaultdict(list)

            # Encoding
            # encoder_out: (patch_seq, batch_size, d_model)
            # encoder_out_dict: [layer] = (patch_seq, batch_size, d_model)
            encoder_out = model.patch_embedding(img).transpose(0, 1) 
            if model.parallel:
                for i in range(len(model.encoders)):
                    encoder_out_dict[i] = model.encoders[i](encoder_out)
            else:
                for i in range(len(model.encoders)):
                    encoder_out = model.encoders[i](encoder_out)

            # Expanding
            # encoder_out: (patch_seq, batch_size * k, d_model)
            # encoder_out_dict: [layer] = (patch_seq, batch_size * k, d_model)
            patch_seq_size = encoder_out.size(0)
            if model.parallel:
                for i in encoder_out_dict:
                    encoder_out_dict[i] = encoder_out_dict[i].view(
                        -1, args.test_batch_size, 1, args.d_model).repeat(1, 1, args.beam_size, 1)
                    encoder_out_dict[i] = encoder_out_dict[i].view(patch_seq_size, -1, args.d_model)
            else:
                encoder_out = encoder_out.view(
                    -1, args.test_batch_size, 1, args.d_model).repeat(1, 1, args.beam_size, 1)
                encoder_out = encoder_out.view(patch_seq_size, -1, args.d_model)

            # Scores save vector & decoding list setting
            scores_save = torch.zeros(args.beam_size * args.test_batch_size, 1, device=device)
            top_k_scores = torch.zeros(args.beam_size * args.test_batch_size, 1, device=device)
            complete_seqs = dict()
            complete_ind = set()

            # Decoding start token setting
            # seqs: (beam_size * batch_size, 1)
            seqs = torch.tensor([[args.bos_id]], dtype=torch.long, device=device) 
            seqs = seqs.repeat(args.beam_size * args.test_batch_size, 1).contiguous()

            for step in range(args.max_len):
                # Decoder setting
                # tgt_mask: (out_seq)
                # tgt_key_padding_mask: (batch_size * k, out_seq)
                tgt_mask = tgt_masks[seqs.size(1)]
                tgt_key_padding_mask = (seqs == model.pad_id)

                # Decoding sentence
                # decoder_out: (out_seq, batch_size * k, d_model)
                decoder_out = model.text_embedding(seqs).transpose(0, 1)
                if model.parallel:
                    for i in range(len(model.decoders)):
                        decoder_out = model.decoders[i](decoder_out, encoder_out_dict[i], tgt_mask=tgt_mask, 
                                        tgt_key_padding_mask=tgt_key_padding_mask)
                else:
                    for i in range(len(model.decoders)):
                        decoder_out = model.decoders[i](decoder_out, encoder_out, tgt_mask=tgt_mask, 
                                        tgt_key_padding_mask=tgt_key_padding_mask)

                # Score calculate
                # scores: (batch_size * k, vocab_num)
                scores = F.gelu(model.trg_output_linear(decoder_out[-1]))
                scores = model.trg_output_linear2(model.trg_output_norm(scores))
                scores = F.log_softmax(scores, dim=1) 

                # Repetition Penalty
                if step > 0 and args.repetition_penalty > 0:
                    prev_ix = next_word_inds.view(-1)
                    for index, prev_token_id in enumerate(prev_ix):
                        scores[index][prev_token_id] *= args.repetition_penalty

                # Add score
                scores = top_k_scores.expand_as(scores) + scores 
                if step == 0:
                    # scores: (batch_size, vocab_num)
                    # top_k_scores: (batch_size, k)
                    scores = scores[::args.beam_size] 
                    scores[:, args.eos_id] = float('-inf') # set eos token probability zero in first step
                    top_k_scores, top_k_words = scores.topk(args.beam_size, 1, True, True)
                else:
                    # top_k_scores: (batch_size * k, out_seq)
                    top_k_scores, top_k_words = scores.view(
                        args.test_batch_size, -1).topk(args.beam_size, 1, True, True)

                # Previous and Next word extract
                # seqs: (batch_size * k, out_seq + 1)
                prev_word_inds = top_k_words // args.vocab_size
                next_word_inds = top_k_words % args.vocab_size
                top_k_scores = top_k_scores.view(args.test_batch_size * args.beam_size, -1)
                top_k_words = top_k_words.view(args.test_batch_size * args.beam_size, -1)
                seqs = seqs[prev_word_inds.view(-1) + every_batch.unsqueeze(1).repeat(1, args.beam_size).view(-1)]
                seqs = torch.cat([seqs, next_word_inds.view(args.beam_size * args.test_batch_size, -1)], dim=1) 

                # Find and Save Complete Sequences Score
                eos_ind = torch.where(next_word_inds.view(-1) == args.eos_id)[0]
                if len(eos_ind) > 0:
                    eos_ind = eos_ind.tolist()
                    complete_ind_add = set(eos_ind) - complete_ind
                    complete_ind_add = list(complete_ind_add)
                    complete_ind.update(eos_ind)
                    if len(complete_ind_add) > 0:
                        scores_save[complete_ind_add] = top_k_scores[complete_ind_add]
                        for ix in complete_ind_add:
                            complete_seqs[ix] = seqs[ix].tolist()

            # If eos token doesn't exist in sequence
            score_save_pos = torch.where(scores_save == 0)
            if len(score_save_pos[0]) > 0:
                for ix in score_save_pos[0].tolist():
                    complete_seqs[ix] = seqs[ix].tolist()
                scores_save[score_save_pos] = top_k_scores[score_save_pos]

            # Beam Length Normalization
            length_penalty = torch.tensor(
                [len(complete_seqs[i]) for i in range(args.test_batch_size * args.beam_size)], device=device)
            length_penalty = (((length_penalty + args.beam_size) ** args.beam_alpha) / ((args.beam_size + 1) ** args.beam_alpha))
            scores_save = scores_save / length_penalty.unsqueeze(1)

            # Predicted and Label processing
            ind = scores_save.view(args.test_batch_size, args.beam_size, -1).argmax(dim=1)
            ind_expand = ind.view(-1) + every_batch
            predicted_seqs = [complete_seqs[i] for i in ind_expand.tolist()]
            predicted_list.extend(predicted_seqs)

            # Print progress
            if print_step % args.print_freq == 0:
                write_log(logger, f'[{print_step}/{len(dataloader_test)}] spend_time: {round((time.time() - start_time) / 60, 3)}min')

            # Results save
            # 1) txt file
            write_mode = 'w' if i == 0 else 'a'
            label_txt = open(os.path.join(args.captioning_save_path, 'label_text.txt'), write_mode)
            predict_txt = open(os.path.join(args.captioning_save_path, 'predict_text.txt'), write_mode)
            for i in range(args.test_batch_size):
                label_txt.write(spm_model.DecodeIds(caption_list[i]) + '\n')
                predict_txt.write(spm_model.DecodeIds(predicted_seqs[i]) + '\n')
            label_txt.close()
            predict_txt.close()

            # 2) Image file
            if print_step % args.print_freq == 0:
                un_norm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                fig = plt.figure(figsize=(64, 64))
                grid_count = ceil(sqrt(args.test_batch_size))
                figure_dict = dict()

                for count in range(1, args.test_batch_size + 1):
                    figure_dict[f'ax{count}'] = fig.add_subplot(grid_count, grid_count, count)
                    new_img = un_norm(img[count-1]).cpu()
                    figure_dict[f'ax{count}'].imshow(new_img.permute(1, 2, 0))
                    xlabel_text = f'''label: {spm_model.DecodeIds(caption_list[count-1])} \n predict: {spm_model.DecodeIds(predicted_seqs[count-1])}'''
                    figure_dict[f'ax{count}'].set_xlabel(xlabel_text, fontsize = 30)
                fig.savefig('./full_figure.jpg')

    with open(f'./results_beam_{args.beam_size}_{args.beam_alpha}_{args.repetition_penalty}.pkl', 'wb') as f:
        pickle.dump({
            'prediction': predicted_list, 
            'label': label_list,
            'prediction_decode': [spm_model.DecodeIds(pred) for pred in predicted_list],
            'label_decode': [spm_model.DecodeIds(label) for label in label_list]
        }, f)