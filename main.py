# Import modules
import time
import argparse
# Training
from task.train_vit import vit_training
from task.train_cap import captioning_training
# from train_transgan import transgan_training
# Testing
from task.test_cap import captioning_testing
from task.train_detr import detr_training
# Utils
from utils import str2bool

def main(args):
    # Time setting
    total_start_time = time.time()

    if args.model == 'ViT':
        if args.training:
            vit_training(args)
        # if args.testing:
        #     vit_testing(args)

    if args.model == 'Captioning':
        if args.training:
            captioning_training(args)
        if args.testing:
            captioning_testing(args)

    # if args.model == 'TransGAN':
    #     if args.training:
    #         transgan_training(args)
    #     if args.testing:
    #         transgan_testing(args)'

    if args.model == 'DETR':
        if args.training:
            detr_training(args)
        # if args.testing:
        #     detr_testing(args)

    # Time calculate
    print(f'Done! ; {round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')
    # Task setting
    parser.add_argument('--model', type=str, choices=['ViT', 'Captioning', 'TransGAN'], required=True,
                        help="Choose model in 'ViT', 'Captioning', 'TransGAN'")
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    # Path setting
    parser.add_argument('--vit_preprocess_path', default='./preprocessing', type=str,
                        help='Pre-processed data save path')
    parser.add_argument('--vit_data_path', default='/HDD/dataset/imagenet/ILSVRC', type=str,
                        help='Original data path')
    parser.add_argument('--vit_save_path', default='/HDD/kyohoon/model_checkpoint/vit/', type=str,
                        help='Model checkpoint file path')
    parser.add_argument('--captioning_preprocess_path', default='./preprocessing', type=str,
                        help='Pre-processed data save path')
    parser.add_argument('--captioning_data_path', default='/HDD/dataset/coco', type=str,
                        help='Original data path')
    parser.add_argument('--captioning_save_path', default='/HDD/kyohoon/model_checkpoint/captioning/', type=str,
                        help='Model checkpoint file path')
    parser.add_argument('--transgan_preprocess_path', default='./preprocessing', type=str,
                        help='Pre-processed data save path')
    parser.add_argument('--transgan_data_path', default='/HDD/dataset/coco', type=str,
                        help='Original data path')
    parser.add_argument('--transgan_save_path', default='/HDD/kyohoon/model_checkpoint/', type=str,
                        help='Model checkpoint file path')
    # Data setting
    parser.add_argument('--img_size', default=256, type=int,
                        help='Image resize size; Default is 256')
    parser.add_argument('--vocab_size', default=8000, type=int,
                        help='Caption vocabulary size; Default is 8000')
    parser.add_argument('--pad_id', default=0, type=int,
                        help='Padding token index; Default is 0')
    parser.add_argument('--unk_id', default=3, type=int,
                        help='Unknown token index; Default is 3')
    parser.add_argument('--bos_id', default=1, type=int,
                        help='Start token index; Default is 1')
    parser.add_argument('--eos_id', default=2, type=int,
                        help='End token index; Default is 2')
    parser.add_argument('--min_len', default=4, type=int,
                        help='Minimum length of caption; Default is 4')
    parser.add_argument('--max_len', default=300, type=int,
                        help='Maximum length of caption; Default is 300')
    # Model setting
    parser.add_argument('--parallel', default=False, type=str2bool,
                        help='Transformer Encoder and Decoder parallel mode; Default is False')
    parser.add_argument('--triple_patch', default=False, type=str2bool,
                        help='Triple patch testing; Default is False')
    parser.add_argument('--patch_size', default=32, type=int, 
                        help='ViT patch size; Default is 32')
    parser.add_argument('--d_model', default=1024, type=int, 
                        help='Transformer model dimension; Default is 768')
    parser.add_argument('--d_embedding', default=256, type=int, 
                        help='Transformer embedding word token dimension; Default is 256')
    parser.add_argument('--n_head', default=16, type=int, 
                        help="Multihead Attention's head count; Default is 16")
    parser.add_argument('--dim_feedforward', default=2048, type=int, 
                        help="Feedforward network's dimension; Default is 2048")
    parser.add_argument('--dropout', default=0.1, type=float, 
                        help="Dropout ration; Default is 0.1")
    parser.add_argument('--embedding_dropout', default=0.1, type=float, 
                        help="Embedding dropout ration; Default is 0.1")
    parser.add_argument('--num_encoder_layer', default=12, type=int, 
                        help="Number of encoder layers; Default is 12")
    parser.add_argument('--num_decoder_layer', default=12, type=int, 
                        help="Number of decoder layers; Default is 12")
    # Optimizer & LR_Scheduler setting
    optim_list = ['AdamW', 'Adam', 'SGD', 'Ralamb']
    scheduler_list = ['constant', 'warmup', 'reduce_train', 'reduce_valid', 'lambda']
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=optim_list,
                        help="Choose optimizer setting in 'AdamW', 'Adam', 'SGD'; Default is AdamW")
    parser.add_argument('--scheduler', default='constant', type=str, choices=scheduler_list,
                        help="Choose optimizer setting in 'constant', 'warmup', 'reduce'; Default is constant")
    parser.add_argument('--n_warmup_epochs', default=2, type=float, 
                        help='Wamrup epochs when using warmup scheduler; Default is 2')
    parser.add_argument('--lr_lambda', default=0.95, type=float,
                        help="Lambda learning scheduler's lambda; Default is 0.95")
    # Training setting
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Num CPU Workers; Default is 8')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size; Default is 16')
    parser.add_argument('--num_epochs', default=100, type=int, 
                        help='Epoch count; Default is 100')
    parser.add_argument('--lr', default=5e-5, type=float,
                        help='Maximum learning rate of warmup scheduler; Default is 5e-5')
    parser.add_argument('--w_decay', default=1e-5, type=float,
                        help="Ralamb's weight decay; Default is 1e-5")
    parser.add_argument('--clip_grad_norm', default=5, type=int, 
                        help='Graddient clipping norm; Default is 5')
    # Testing setting
    parser.add_argument('--test_batch_size', default=32, type=int, 
                        help='Test batch size; Default is 32')
    parser.add_argument('--beam_size', default=5, type=int, 
                        help='Beam search size; Default is 5')
    parser.add_argument('--beam_alpha', default=0.7, type=float, 
                        help='Beam search length normalization; Default is 0.7')
    parser.add_argument('--repetition_penalty', default=1.3, type=float, 
                        help='Beam search repetition penalty term; Default is 1.3')
    # Print frequency
    parser.add_argument('--print_freq', default=100, type=int, 
                        help='Print training process frequency; Default is 100')


    ######################## DETR #########################
    # python main.py --model DETR --lr 1e-4 --batch_size 2 --weight_decay 1e-4 --num_epochs 300 --num_encoder_layer 6 --num_decoder_layer 6 
    # --dim_feedforward 2048 --d_model 256 --dropout 0.1 --n_head 8 
    
    parser.add_argument('--lr_backbone', default=1e-5, type=float)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    # parser.add_argument('--pre_norm', action='store_false')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    
    main(args)