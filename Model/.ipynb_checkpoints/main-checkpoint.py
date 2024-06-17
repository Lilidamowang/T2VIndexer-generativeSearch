import os
import argparse
import pickle

import nltk
import pandas as pd
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from main_utils import set_seed
from main_models import T5FineTuner
from pytorch_lightning.plugins import DDPPlugin

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

def train(args):
    model = T5FineTuner(args)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.output_dir,
        filename=args.tag_info + '_{epoch}-{recall1:.6f}',
        monitor="recall1",
        save_on_train_epoch_end=False,
        mode="max",
        save_top_k=1,
        every_n_val_epochs=args.check_val_every_n_epoch,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor()
    # 设置epoch
    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        precision=16 if args.fp_16 else 32, #  precision="bf16"
        amp_level=args.opt_level,
        resume_from_checkpoint=args.resume_from_checkpoint,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=True,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        logger=logger,
        callbacks=[lr_monitor, checkpoint_callback],
        plugins=DDPPlugin(find_unused_parameters=False),
        accelerator=args.accelerator,
        amp_backend='apex',
    )
    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

def parsers_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--model_name_or_path', type=str, default="t5-")
    parser.add_argument('--tokenizer_name_or_path', type=str, default="t5-")
    parser.add_argument('--freeze_encoder', type=int, default=0, choices=[0, 1])
    parser.add_argument('--freeze_embeds', type=int, default=0, choices=[0, 1])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--num_train_epochs', type=int, default=500)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--n_val', type=int, default=-1)
    parser.add_argument('--n_train', type=int, default=-1)
    parser.add_argument('--n_test', type=int, default=-1)
    parser.add_argument('--early_stop_callback', type=int, default=0, choices=[0, 1])
    parser.add_argument('--fp_16', type=int, default=0, choices=[0, 1])
    parser.add_argument('--opt_level', type=str, default='O1')
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrain_encoder', type=int, default=1, choices=[0, 1])
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    parser.add_argument('--softmax', type=int, default=0, choices=[0, 1])
    parser.add_argument('--aug', type=int, default=0, choices=[0, 1])
    parser.add_argument('--accelerator', type=str, default="ddp")
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_cls', type=int, default=1000)
    parser.add_argument('--decode_embedding', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--output_vocab_size', type=int, default=30)
    parser.add_argument('--hierarchic_decode', type=int, default=0, choices=[0, 1])
    parser.add_argument('--tie_word_embedding', type=int, default=0, choices=[0, 1])
    parser.add_argument('--tie_decode_embedding', type=int, default=1, choices=[0, 1])
    parser.add_argument('--gen_method', type=str, default="greedy")
    parser.add_argument('--length_penalty', type=int, default=0.8)

    parser.add_argument('--recall_num', type=list, default=[1,5,10,20,50,100], help='[1,5,10,20,50,100]')
    parser.add_argument('--random_gen', type=int, default=0, choices=[0, 1])
    parser.add_argument('--label_length_cutoff', type=int, default=0)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--val_check_interval', type=float, default=1.0)
    
    parser.add_argument('--test_set', type=str, default="dev")
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=2)
    
    parser.add_argument('--max_input_length', type=int, default=40)
    parser.add_argument('--inf_max_input_length', type=int, default=40)
    parser.add_argument('--max_output_length', type=int, default=10)
    parser.add_argument('--doc_length', type=int, default=64)
    parser.add_argument('--contrastive_variant', type=str, default="", help='E_CL, ED_CL, doc_Reweight')
    parser.add_argument('--num_return_sequences', type=int, default=100, help='generated id num (include invalid)')
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'eval', 'calculate'])
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--decoder_learning_rate', type=float, default=1e-4)
    parser.add_argument('--certain_epoch', type=int, default=None)
    parser.add_argument('--given_ckpt', type=str, default='')
    parser.add_argument('--infer_ckpt', type=str, default='')
    parser.add_argument('--model_info', type=str, default='base', choices=['small', 'large', 'base', '3b', '11b'])
    parser.add_argument('--id_class', type=str, default='k30_c30')
    parser.add_argument('--ckpt_monitor', type=str, default='recall', choices=['recall', 'train_loss'])
    parser.add_argument('--Rdrop', type=float, default=0.15, help='default to 0-0.3')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--Rdrop_only_decoder', type=int, default=0,
                        help='1-RDrop only for decoder, 0-RDrop only for all model', choices=[0,1])
    parser.add_argument('--Rdrop_loss', type=str, default='KL', choices=['KL', 'L2'])
    parser.add_argument('--adaptor_decode', type=int, default=1, help='default to 0,1')
    parser.add_argument('--adaptor_efficient', type=int, default=1, help='default to 0,1')
    parser.add_argument('--adaptor_layer_num', type=int, default=4)
    parser.add_argument('--test1000', type=int, default=0, help='default to 0,1')
    parser.add_argument('--position', type=int, default=1)
    parser.add_argument('--contrastive', type=int, default=0)
    parser.add_argument('--embedding_distillation', type=float, default=0.0)
    parser.add_argument('--weight_distillation', type=float, default=0.0)
    parser.add_argument('--hard_negative', type=int, default=0)
    parser.add_argument('--aug_query', type=int, default=1)
    parser.add_argument('--aug_query_type', type=str, default='corrupted_query', help='aug_query, corrupted_query')
    parser.add_argument('--sample_neg_num', type=int, default=0)
    parser.add_argument('--query_tloss', type=int, default=0)
    parser.add_argument('--weight_tloss', type=int, default=0)
    parser.add_argument('--ranking_loss', type=int, default=0)
    parser.add_argument('--disc_loss', type=int, default=0)
    parser.add_argument('--input_dropout', type=int, default=1)
    parser.add_argument('--denoising', type=int, default=0)
    parser.add_argument('--multiple_decoder', type=int, default=0)
    parser.add_argument('--decoder_num', type=int, default=1)
    parser.add_argument('--loss_weight', type=int, default=0)
    parser.add_argument('--kary', type=int, default=30)
    parser.add_argument('--tree', type=int, default=1)
    parser.add_argument('--info', type=str, default='')

    parser_args = parser.parse_args()

    # args post process
    parser_args.tokenizer_name_or_path += parser_args.model_info
    parser_args.model_name_or_path += parser_args.model_info

    parser_args.gradient_accumulation_steps = max(int(8 / parser_args.n_gpu), 1)


    if parser_args.mode == 'train':
        # set to small val to prevent CUDA OOM
        parser_args.num_return_sequences = 10
        parser_args.eval_batch_size = 1

    if parser_args.model_info == 'base':
        parser_args.num_layers = 12
        parser_args.num_decoder_layers = 6
        parser_args.d_ff = 3072
        parser_args.d_model = 768
        parser_args.num_heads = 12
        parser_args.d_kv = 64
    elif parser_args.model_info == 'large':
        parser_args.num_layers = 24
        parser_args.num_decoder_layers = 12
        parser_args.d_ff = 4096
        parser_args.d_model = 1024
        parser_args.num_heads = 16
        parser_args.d_kv = 64
    elif parser_args.model_info == 'small':
        parser_args.num_layers = 6
        parser_args.num_decoder_layers = 3
        parser_args.d_ff = 2048
        parser_args.d_model = 512
        parser_args.num_heads = 8
        parser_args.d_kv = 64

    if parser_args.test1000:
        parser_args.n_val = 1000
        parser_args.n_train = 1000
        parser_args.n_test = 1000

    return parser_args

if __name__ == "__main__":
    args = parsers_parser()
    set_seed(args.seed)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(dir_path)
    # print(parent_path)
    args.logs_dir = dir_path + '/logs/'
    
    # # this is model pkl save dir
    args.output_dir = dir_path + '/logs/'

    time_str = time.strftime("%Y%m%d-%H%M%S")
    # Note -- you can put important info into here, then it will appear to the name of saved ckpt
    important_info_list = ["kary:", str(args.kary), args.model_info, args.id_class,
                           args.test_set, args.ckpt_monitor, 'dem:',
                           str(args.decode_embedding), 'ada:', str(args.adaptor_decode), 'adaeff:',
                           str(args.adaptor_efficient), 'adanum:', str(args.adaptor_layer_num), 'RDrop:', str(args.dropout_rate), str(args.Rdrop), str(args.Rdrop_only_decoder)]

    args.query_info = '_'.join(important_info_list)
    # if YOUR_API_KEY != '':
    #     os.environ["WANDB_API_KEY"] = YOUR_API_KEY
    #     logger = WandbLogger(name='{}-{}'.format(time_str, args.query_info), project='l1-t5-nq')
    # else:
    logger = TensorBoardLogger("logs/")
    # ###########################

    args.tag_info = '{}_lre{}d{}'.format(args.query_info, str(float(args.learning_rate * 1e4)),
                                                           str(float(args.decoder_learning_rate * 1e4)))
    args.res1_save_path = args.logs_dir + '{}_res1_recall{}_{}.tsv'.format(
         args.tag_info, args.num_return_sequences, time_str)

    train(args)