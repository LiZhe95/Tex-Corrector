# -*- coding: utf-8 -*-
"""
@Time  :2023/3/24 14:22
@File  :data_process.py
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader

from utils.file_utils import load_json, load_txt


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class BertDataCollator(DataCollator):
    def __init__(self, tokenizer):
        super(BertDataCollator, self).__init__(tokenizer)

    # def __check(self, ori, cor):
    #      _ori = self.tokenizer(ori, padding=True, return_tensors='pt')
    #      _cor = self.tokenizer(cor, padding=True, return_tensors='pt')['input_ids']
    #     if _ori['input_ids'].shape != _cor.shape:
    #         return None, None
    #     else:
    #         return _ori, _cor

    def __call__(self, data, *args, **kwargs):
        ori_text, cor_text = zip(*data)
        # ori_token, cor_token = self.__check(ori_text, cor_text)
        return ori_text, cor_text


class MacBertDataCollator(DataCollator):
    def __init__(self, tokenizer):
        super(MacBertDataCollator, self).__init__(tokenizer)

    def __call__(self, data):
        ori_texts, cor_texts, wrong_idss = zip(*data)
        if len(ori_texts) != len(cor_texts):
            raise ValueError(f"ori_texts must be equal cor_texts, ori_texts is:{ori_texts},cor_texts is:{cor_texts}")
        encoded_texts = [self.tokenizer.tokenize(t) for t in ori_texts]
        max_len = max([len(t) for t in encoded_texts]) + 2
        det_labels = torch.zeros(len(ori_texts), max_len).long()
        for i, (encoded_text, wrong_ids) in enumerate(zip(encoded_texts, wrong_idss)):
            for idx in wrong_ids:
                margins = []
                for word in encoded_text[:idx]:
                    if word == '[UNK]':
                        break
                    if word.startswith('##'):
                        margins.append(len(word) - 3)
                    else:
                        margins.append(len(word) - 1)
                margin = sum(margins)
                move = 0
                while (abs(move) < margin) or (idx + move >= len(encoded_text)) \
                        or encoded_text[idx + move].startswith('##'):
                    move -= 1
                det_labels[i, idx + move + 1] = 1
        return ori_texts, cor_texts, det_labels


class CorrectorDataset(Dataset):
    def __init__(self, fp, model_name):
        self.model_name = model_name
        if self.model_name == 'macbert':
            self.data = load_json(fp)
        elif self.model_name == 'bert':
            self.data = load_txt(fp)
        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.model_name == 'macbert':
            return self.data[index]['original_text'], \
                   self.data[index]['correct_text'], self.data[index]['wrong_ids']
        elif self.model_name == 'bert':
            data = self.data[index].strip().split('\t')
            if len(data) != 2:
                return None
            else:
                return data


def get_corrector_loader(collate_fn, model_name, path, batch_size, shuffle, num_workers, **kwargs):

    if path and os.path.exists(path):
        loader = DataLoader(
                            CorrectorDataset(path, model_name),
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    else:
        raise FileExistsError(f'{path} is not exit')

    return loader

