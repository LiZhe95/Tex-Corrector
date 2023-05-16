# -*- coding: utf-8 -*-
"""
@Time  :2023/3/24 14:00
@File  :main.py
"""

import os
import torch
import random
import pytorch_lightning as pl

from collections import OrderedDict
from transformers import BertTokenizer, BertForMaskedLM


from config.config import parse_args
from utils.logger import get_logger
from data_process import BertDataCollator, MacBertDataCollator
from data_process import get_corrector_loader
from models.bert.bert_csc import Bert4Csc
from models.macbert.macbert4csc import MacBert4Csc
from models.macbert_kl.macbert4csc import MacBert4Csc as MacBert4CscKL
from pytorch_lightning.strategies.ddp import DDPStrategy


def seed_torch(seed=42):    # 随机数种子
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)    # 为了禁止hash随机化，使得实验可复现
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    # 加载tokenizer
    args = parse_args()
    logger = get_logger('text_corrector', log_file=args.log_path)
    logger.info('traing params:{}'.format(args.__dict__))
    args.logger = logger

    tokenizer = BertTokenizer.from_pretrained(args.bert_checkpoint)
    if args.model_name == 'bert':
        model = Bert4Csc(args, tokenizer)
        collator = BertDataCollator(tokenizer=tokenizer)

    elif args.model_name == 'macbert':
        model = MacBert4Csc(args, tokenizer)
        collator = MacBertDataCollator(tokenizer=tokenizer)

    elif args.model_name == 'macbert_kl':
        model = MacBert4CscKL(args, tokenizer, training=True)
        collator = MacBertDataCollator(tokenizer=tokenizer)
        args.model_name = 'macbert'

    else:
        raise ValueError('model name is not support')

    for name, param in model.named_parameters():
        if 'bert_long_trem' in name:
            param.requires_grad = False
    logger.info('load tokenizer and model success')
    # 加载数据
    train_loader = get_corrector_loader(collator,
                                        args.model_name,
                                        args.train_path,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=4)
    if 'dev' in args.mode:
        valid_loader = get_corrector_loader(collator,
                                            args.model_name,
                                            args.dev_path,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4)
    if 'test' in args.mode:
        test_loader = get_corrector_loader(collator,
                                           args.model_name,
                                           args.test_path,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=4)
    logger.info('load datas success')

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.output_dir,
        filename='{epoch:02d}-{val_loss:.4f}',
        save_top_k=-1,
        mode='min'
    )
    trainer = pl.Trainer(
                         logger=False,
                         max_epochs=args.epochs,
                         # strategy = DDPStrategy(find_unused_parameters=True)
                         accelerator="cpu" if args.device == 'cpu' else 'gpu',
                         devices=[int(i) for i in args.gpu_ids.split(',')],
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         callbacks=[ckpt_callback])

    # 进行训练
    # train_loader中有数据
    # 正向传播时：开启自动求导的异常侦测
    torch.autograd.set_detect_anomaly(True)
    if 'train' in args.mode and train_loader and len(train_loader) > 0:
        if valid_loader and len(valid_loader) > 0:
            trainer.fit(model, train_loader, valid_loader)
        else:
            trainer.fit(model, train_loader)
        logger.info('train model done.')

    # 模型转为transformers可加载
    if ckpt_callback and len(ckpt_callback.best_model_path) > 0:
        ckpt_path = ckpt_callback.best_model_path
    else:
        ckpt_path = ''

    if ckpt_path and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        # 先保存原始transformer bert model
        tokenizer.save_pretrained(args.output_dir)
        bert = BertForMaskedLM.from_pretrained(args.bert_checkpoint)
        bert.save_pretrained(args.output_dir)
        state_dict = torch.load(ckpt_path)['state_dict']

        new_state_dict = OrderedDict()
        if args.model_name in ['bert', 'macbert', 'macbert_kl']:
            for k, v in state_dict.items():
                if k.startswith('bert.'):
                    new_state_dict[k[5:]] = v
        else:
            new_state_dict = state_dict
        # 再保存finetune训练后的模型文件，替换原始的pytorch_model.bin
        torch.save(new_state_dict, os.path.join(args.output_dir, 'pytorch_model.bin'))
        logger.info('train model save done')
    # 进行测试的逻辑同训练
    if 'test' in args.mode and test_loader and len(test_loader) > 0:
        logger.info('test begin')
        trainer.test(model, test_loader)
        logger.info('test done')


if __name__ == '__main__':
    main()

