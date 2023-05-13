# -*- coding: utf-8 -*-
"""
@Time  :2023/3/24 14:29
@File  :base_model.py
"""

import operator
from abc import ABC
import logging
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from models.bert import lr_scheduler
from models.bert.evaluate_util import compute_corrector_prf, compute_sentence_level_prf
from utils.logger import get_logger

logger = get_logger('bert', './log/train.log')

class FocalLoss(nn.Module):
    """
    Softmax and sigmoid focal loss.
    copy from https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, num_labels, activation_type='softmax', gamma=2.0, alpha=0.25, epsilon=1.e-9):

        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_type = activation_type

    def forward(self, input, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(input, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = torch.sigmoid(input)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.lr
        weight_decay = cfg.weight_decay
        if "bias" in key:
            lr = cfg.base_lr * cfg.bias_lr_factor
            weight_decay = cfg.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.optimizer_name == 'SGD':
        optimizer = getattr(torch.optim, cfg.optimizer_name)(params, momentum=cfg.momentum)
    else:
        optimizer = getattr(torch.optim, cfg.optimizer_name)(params)
    return optimizer


def build_lr_scheduler(cfg, optimizer):
    scheduler_args = {
        "optimizer": optimizer,

        # warmup options
        "warmup_factor": cfg.warmup_factor,
        "warmup_epochs": cfg.warmup_epochs,
        "warmup_method": cfg.warmup_method,

        # multi-step lr scheduler options
        "milestones": cfg.steps,
        "gamma": cfg.gamma,

        # cosine annealing lr scheduler options
        "max_iters": cfg.max_iter,
        "delay_iters": cfg.delay_iters,
        "eta_min_lr": cfg.eta_min_lr,

    }
    scheduler = getattr(lr_scheduler, cfg.sched)(**scheduler_args)
    return {'scheduler': scheduler, 'interval': cfg.interval}


class BaseTrainingEngine(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def configure_optimizers(self):
        optimizer = make_optimizer(self.cfg, self)
        scheduler = build_lr_scheduler(self.cfg, optimizer)

        return [optimizer], [scheduler]

    def on_validation_epoch_start(self) -> None:
        self.cfg.logger.info('Valid.')

    def on_test_epoch_start(self) -> None:
        self.cfg.logger.info('Testing...')


class CscTrainingModel(BaseTrainingEngine, ABC):
    """
        用于CSC的BaseModel, 定义了训练及预测步骤
        """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # loss weight
        self.w = cfg.loss_weight
        self.valid_loss = []
        self.valid_cor_acc_labels = []
        self.valid_result = []

    def training_step(self, batch, batch_idx):
        ori_text, cor_text = batch
        outputs = self.forward(ori_text, cor_text)
        loss = outputs[0]
        # if loss:
            # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(ori_text))
        return loss

    def validation_step(self, batch, batch_idx):
        ori_text, cor_text = batch
        outputs = self.forward(ori_text, cor_text)
        loss = outputs[0]
        if loss:
            cor_y_hat = torch.argmax((outputs[1]), dim=-1)
            encoded_x = self.tokenizer(cor_text, padding=True, return_tensors='pt')
            encoded_x.to(self._device)
            cor_y = encoded_x['input_ids']
            cor_y_hat *= encoded_x['attention_mask']

            results = []
            cor_acc_labels = []
            for src, tgt, predict in zip(ori_text, cor_y, cor_y_hat):
                _src = self.tokenizer(src, add_special_tokens=False)['input_ids']
                _tgt = tgt[1:len(_src) + 1].cpu().numpy().tolist()
                _predict = predict[1:len(_src) + 1].cpu().numpy().tolist()
                cor_acc_labels.append(1 if operator.eq(_tgt, _predict) else 0)
                results.append((_src, _tgt, _predict,))
                self.valid_cor_acc_labels += cor_acc_labels
                self.valid_result += results
            self.valid_loss.append(loss.cpu().item())

        # return loss.cpu().item(), cor_acc_labels, results


    def on_validation_epoch_end(self) -> None:
        loss = np.mean(self.valid_loss)
        self.log('val_loss', loss)
        self.cfg.logger.info(f'loss: {loss}')
        self.cfg.logger.info(f'Correction:\n'
                    f'acc: {np.mean(self.valid_cor_acc_labels):.4f}')
        compute_corrector_prf(self.valid_result, logger)
        compute_sentence_level_prf(self.valid_result, logger)
        self.valid_loss.clear()
        self.valid_cor_acc_labels.clear()
        self.valid_result.clear()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        logger.info('Test.')
        self.on_validation_epoch_end()

    def predict(self, texts, device):
        inputs = self.tokenizer(texts, padding=True, return_tensors='pt')
        with torch.no_grad():
            inputs.to(device)
            outputs = self.forward(texts)
            y_hat = torch.argmax(outputs, dim=-1)
            expand_text_lens = torch.sum(inputs['attention_mask'], dim=-1) - 1

        rst = []
        for t_len, _y_hat in zip(expand_text_lens, y_hat):
            rst.append(self.tokenizer.decode(_y_hat[1:t_len]).replace(' ', ''))
        return rst
