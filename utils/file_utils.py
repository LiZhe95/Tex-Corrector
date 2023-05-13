# -*- coding: utf-8 -*-
"""
@Time  :2023/3/24 14:56
@File  :file_utils.py
"""
import json


def load_json(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_txt(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        return f.readlines()



