# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Abtion(abtion@outlook.com)
@description: 
"""
import sys
sys.path.append('../')
from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM
from models.macbert_kl.base_model import CscTrainingModel, FocalLoss
import copy


class MacBert4Csc(CscTrainingModel, ABC):
    def __init__(self, cfg, tokenizer, training=False):
        super().__init__(cfg)
        self.cfg = cfg
        self.bert = BertForMaskedLM.from_pretrained(cfg.bert_checkpoint)
        self.detection = nn.Linear(self.bert.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = tokenizer
        if training:
            self.bert_long_term = copy.deepcopy(self.bert)
        self.eopch_first = False
        self.now_epoch = 0

    def forward(self, texts, cor_labels=None, det_labels=None):
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
        # 检错概率
        prob = self.detection(bert_outputs.hidden_states[-1])

        if self.current_epoch != self.now_epoch and text_labels is not None:
            self.bert_long_term = copy.deepcopy(self.bert)
            self.now_epoch = self.current_epoch

        if text_labels is None:
            # 检错输出，纠错输出
            outputs = (prob, bert_outputs.logits)
        else:
            # long term output
            bert_outputs_long_term = self.bert_long_term(**encoded_text, labels=text_labels, return_dict=True,
                                                         output_hidden_states=True)

            det_loss_fct = FocalLoss(num_labels=None, activation_type='sigmoid')
            # pad部分不计算损失
            active_loss = encoded_text['attention_mask'].view(-1, prob.shape[1]) == 1
            active_probs = prob.view(-1, prob.shape[1])[active_loss]
            active_labels = det_labels[active_loss]
            det_loss = det_loss_fct(active_probs, active_labels.float())
            # 检错loss，纠错loss，检错输出，纠错输出
            outputs = (det_loss,
                       bert_outputs.loss,
                       self.sigmoid(prob).squeeze(-1),
                       bert_outputs.logits,
                       bert_outputs_long_term.logits.cpu().detach())
        return outputs
