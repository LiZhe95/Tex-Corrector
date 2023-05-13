# 文本纠错
# 支持模型bert、macbert
### python版本
```python3.9.16```
### 安装依赖
```pip install -r requirements.txt```

# Usage
## 前期准备
需要下载bert模型到pretrain_models/bert-base-chinese目录下
https://huggingface.co/bert-base-chinese/tree/main
## 数据格式
### bert
```
# error_text\tcor_text\n
你说他像窝瓜吧有身子有眼儿的 你说他像窝瓜吧有鼻子有眼儿的
家长一个一个接孩子都有于读的 家长一个一个接孩子都有手续的
鼻子下巴眼睛头发耳杂 鼻子下巴眼睛头发耳朵
.....

```
### macbert
```
[
    {
        "id":"A2-0011-1",
        "original_text":"你好！我是张爱文。",
        "wrong_ids":[],
        "correct_text":"你好！我是张爱文。"
    },
    {
        "id":"A2-0023-1",
        "original_text":"下个星期，我跟我朋唷打算去法国玩儿。",
        "wrong_ids":[
            9
        ],
        "correct_text":"下个星期，我跟我朋友打算去法国玩儿。"
    },
    ......
]
```
## train
具体参数配置在config/config.py
```
python train.py
```
## inference
```
python infer.py
```

# Reference
https://github.com/shibing624/pycorrector
