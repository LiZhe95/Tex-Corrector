# -*- coding: utf-8 -*-
"""
@Time  :2023/3/28 10:17
@File  :infer.py
"""
import torch
from transformers import BertTokenizer

from config.config import parse_args
from models.bert.bert_csc import Bert4Csc
from models.macbert.macbert4csc import MacBert4Csc
from models.macbert_kl.macbert4csc import MacBert4Csc as MacBert4CscKL


device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")


class Inference:
    def __init__(self,
                 model_name='bert',
                 vocab_path='./pretrain_models/bert-base-chinese/vocab.txt',
                 ckpt_path='./outputs/bert/epoch=09-val_loss=0.2646.ckpt'):
        args = parse_args()
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)
        if model_name == 'bert':
            self.model = Bert4Csc.load_from_checkpoint(checkpoint_path=ckpt_path,
                                                       cfg=args,
                                                       map_location=device,
                                                       tokenizer=self.tokenizer)
        elif model_name == 'macbert':
            self.model = MacBert4Csc.load_from_checkpoint(checkpoint_path=ckpt_path,
                                                          cfg=args,
                                                          map_location=device,
                                                          tokenizer=self.tokenizer)   
        elif model_name == 'macbert_kl':
            self.model = MacBert4CscKL.load_from_checkpoint(checkpoint_path=ckpt_path,
                                                            cfg=args,
                                                            map_location=device,
                                                            tokenizer=self.tokenizer)

        self.model.to(device)
        self.model.eval()

    def predict(self, sentence_list):
        """
        文本纠错模型预测
        Args:
            sentence_list: list
                输入文本列表
        Returns: tuple
            corrected_texts(list)
        """
        is_str = False
        if isinstance(sentence_list, str):
            is_str = True
            sentence_list = [sentence_list]
        corrected_texts = self.model.predict(sentence_list, device)
        if is_str:
            return corrected_texts[0]
        return corrected_texts


if __name__ == '__main__':
    infer = Inference(model_name='macbert_kl',
                      vocab_path='./pretrain_models/bert-base-chinese/vocab.txt',
                      ckpt_path='./outputs/macbert_kl_SIGHAN_Wang271K/macbert/epoch=09-val_loss=0.0162.ckpt')

    print(infer.predict(['1.配合国产化OA系统上线，重新构建短信发送平台。各处室、单位可根据工作徐要自主发送断信佟知。']))

