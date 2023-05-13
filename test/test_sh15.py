# -*- coding: utf-8 -*-
"""
@Time  :2023/5/13 19:10
@File  :test_sh15.py
"""
import sys
sys.path.append('..')
import unittest

from infer import Inference
from utils.metrics import compute_corrector_prf


class TestSh(unittest.TestCase):
    def __init__(self):
        self.model = Inference(model_name='macbert_kl',
                               vocab_path='./pretrain_models/bert-base-chinese/vocab.txt',
                               ckpt_path='./outputs/macbert_kl_SIGHAN_Wang271K/macbert/epoch=09-val_loss=0.0162.ckpt')

    def test_sh15(self):
        results = []
        with open('../datas/bert/20230325/sh15_test.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split('\t')
                src = line[0]
                tgt = line[1]
                predict = self.model.predict([src][0])
                if 'UNK' in predict:
                    continue
                results.append((src.lower(), tgt.lower(), predict))
                print(src)
                print(tgt)
                print(predict)
                print(predict == tgt)
                print('*' * 100)
        compute_corrector_prf(results)


if __name__ == '__main__':
    unittest.main()
