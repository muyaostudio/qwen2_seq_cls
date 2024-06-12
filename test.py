#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : test.py
@Date    : 2024/06/11 20:32:26
@Author  : muyaostudio
@Version : 1.0
@Desc    : 
'''

from transformers import Qwen2ForSequenceClassification, Qwen2Tokenizer
import torch

# 加载模型和分词器
model_name = "./results/ckpt"
tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
model = Qwen2ForSequenceClassification.from_pretrained(model_name)

# 准备输入文本
texts = [
    "哈哈哈我太高兴了", 
    "听了一首许嵩的歌，心情非常愉快"
    "今天的考试成绩不理想，感觉很失望", 
    "今天的工作任务很繁重，感到压力很大",
    "今天的学习就是复习之前的知识，没有新的内容"
]
# 创建标签到索引的映射
label_to_id = {
    "积极": 0,
    "消极": 1,
    "中立": 2
}

# 对文本进行编码
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# 进行推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测结果
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)

# 打印预测结果
for text, prediction in zip(texts, predictions):
    print(f"文本: {text} -> 预测类别: {list(label_to_id.keys())[prediction.item()]}")
