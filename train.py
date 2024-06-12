#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : train.py
@Date    : 2024/06/11 20:01:24
@Author  : muyaostudio
@Version : 1.0
@Desc    : 
'''

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import json, random
from datasets import Dataset


# 一些参数
model_name = "Qwen/Qwen2-0.5B"  # 模型名或者本地路径
file_path = './data/example.jsonl'  # 定义训练集路径
save_path = "./results/ckpt"

num_train_epochs=10
per_device_train_batch_size=64
per_device_eval_batch_size=64
warmup_steps=50
weight_decay=0.01
logging_steps=1
use_cpu=True

# 创建标签到索引的映射
label_to_id = {
    "积极": 0,
    "消极": 1,
    "中立": 2
}
num_labels = len(label_to_id)  # 根据你的标签数量设置num_labels

# 加载预训练的 Qwen2 模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)  
model.config.pad_token_id = 151643  # 定义pad token，模型才会忽略后面那些pad而是把真正最后一个token的hidden state用于分类

# 读取jsonl文件
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

data = read_jsonl(file_path)
random.shuffle(data)

# 将文本标签转换为数值标签
for example in data:
    example['label'] = label_to_id[example['label']]

# 检查标签范围
for example in data:
    assert 0 <= example['label'] < len(label_to_id), f"Label out of range: {example['label']}"    

# 将数据转换为datasets库的Dataset对象
dataset = Dataset.from_list(data)

# 将数据集拆分为训练集和验证集
dataset = dataset.train_test_split(test_size=0.2)

# 定义一个函数来处理数据集中的文本
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, return_tensors="pt")

# 对数据集进行预处理
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir=save_path,                           # 输出目录
    num_train_epochs=num_train_epochs,              # 训练的epoch数
    per_device_train_batch_size=per_device_train_batch_size,    # 每个设备的训练batch size
    per_device_eval_batch_size=per_device_eval_batch_size,      # 每个设备的评估batch size
    warmup_steps=warmup_steps,                  # 预热步数
    weight_decay=weight_decay,                  # 权重衰减
    logging_dir=save_path,                      # 日志目录
    logging_steps=logging_steps,
    evaluation_strategy="epoch",
    save_strategy="epoch",    # 每个epoch保存一次检查点
    save_total_limit=3,       # 最多保存3个检查点，旧的会被删除
    use_cpu=use_cpu           # 禁用 CUDA，在 CPU 上运行
)

# 定义Trainer
trainer = Trainer(
    model=model,                                    # 模型
    args=training_args,                             # 训练参数
    train_dataset=encoded_dataset['train'],         # 训练数据集
    eval_dataset=encoded_dataset['test']            # 评估数据集
)

# 开始训练
trainer.train()
trainer.save_state()
trainer.save_model(output_dir=save_path)
tokenizer.save_pretrained(save_path)
