# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description:
"""
import sys
sys.path.append('../')
from abc import ABC

import torch.nn as nn
from transformers import BertForMaskedLM
from models.bert.base_model import CscTrainingModel, FocalLoss


class Bert4Csc(CscTrainingModel, ABC):
    def __init__(self, cfg, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.bert = BertForMaskedLM.from_pretrained(cfg.bert_checkpoint)
        self.tokenizer = tokenizer

    def forward(self, texts, cor_labels=None):
        encoded_text = self.tokenizer(texts, padding=True, return_tensors='pt')
        encoded_text.to(self.device)
        if cor_labels:
            text_labels = self.tokenizer(cor_labels, padding=True, return_tensors='pt')['input_ids']
            text_labels[text_labels == 0] = -100  # -100计算损失时会忽略
            text_labels = text_labels.to(self.device)

            if text_labels.shape != encoded_text['input_ids'].shape:
                # return (None, None)
                text_labels = encoded_text['input_ids'].clone()
                text_labels[text_labels == 0] = -100

        else:
            text_labels = None

        bert_outputs = self.bert(**encoded_text, labels=text_labels, return_dict=True, output_hidden_states=True)

        if text_labels is None:
            # 纠错输出
            outputs = bert_outputs.logits
        else:
            # pad部分不计算损失
            # 纠错loss，纠错输出
            outputs = (
                       bert_outputs.loss,
                       bert_outputs.logits)
        return outputs
