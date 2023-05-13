# -*- coding: utf-8 -*-
"""
@Time  :2023/3/24 15:02
@File  :config.py
"""

import os
import torch
import argparse


def parse_args():
    """
    参数解析
    :return:
    """
    parser = argparse.ArgumentParser(description='Chinese Text csc')
    parser.add_argument('--data_name', default='macbert', type=str)
    parser.add_argument('--model_name', default='macbert', type=str)

    parser.add_argument('--data_path', default='./datas')

    parser.add_argument('--train_path', default='SIGHAN_Wang271K/train.json', type=str)
    parser.add_argument('--dev_path', default='SIGHAN_Wang271K/dev.json', type=str)
    parser.add_argument('--test_path', default='SIGHAN_Wang271K/test.json', type=str)

    parser.add_argument('--output_dir', default='./outputs/macbert_SIGHAN_Wang271K', type=str)
    parser.add_argument('--log_path', default='./logs/macbert_SIGHAN_Wang271K_train.log', type=str)
    parser.add_argument('--mode', default='train,dev,test', type=str)

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--gpu_ids', default='3,4,5,6,7,8', type=str)
    parser.add_argument('--bert_checkpoint', default='pretrain_models/bert-base-chinese', type=str)
    parser.add_argument('--dataloader_num_workers', default=4, type=int)

    parser.add_argument('--optimizer_name', default='AdamW', type=str)
    parser.add_argument('--base_lr', default=5e-5, type=float, help='')
    parser.add_argument('--bias_lr_factor', default=2, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--weight_decay_bias', default=0, type=int)
    parser.add_argument('--gamma', default=0.9999, type=int)
    parser.add_argument('--steps', default=(10,), type=tuple)
    parser.add_argument('--sched', default='WarmupExponentialLR', type=str)
    parser.add_argument('--warmup_factor', default=0.01, type=float)
    parser.add_argument('--warmup_method', default='linear', type=str)
    parser.add_argument('--delay_iters', default=0, type=int)
    parser.add_argument('--eta_min_lr', default=3e-7, type=float)
    parser.add_argument('--max_iter', default=10, type=int)
    parser.add_argument('--interval', default='step', type=str)
    parser.add_argument('--checkpoint_period', default=10, type=int)
    parser.add_argument('--log_period', default=100, type=int)
    parser.add_argument('--accumulate_grad_batches', default=4, type=int, help='梯度累加的batch数')

    parser.add_argument('--loss_weight', default=0.3, type=float, help='论文中的lambda，即correction loss的权重')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='学习率')
    parser.add_argument('--warmup_epochs', default=1024, type=int, help='warmup轮数, 需小于训练轮数')
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    args.__setattr__('train_path', os.path.join(args.data_path, args.data_name, args.train_path))
    args.__setattr__('dev_path', os.path.join(args.data_path, args.data_name, args.dev_path))
    args.__setattr__('test_path', os.path.join(args.data_path, args.data_name, args.test_path))
    args.__setattr__('output_dir', os.path.join(args.output_dir, args.data_name))

    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
