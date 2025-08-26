# -*- coding: utf-8 -*-
# 环境安装：pip install transformers torch datasets
import numpy as np
import pandas as pd
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    TrainingArguments, 
    Trainer
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from datasets import load_dataset, Dataset, ClassLabel
from transformers import EarlyStoppingCallback
from transformers import DataCollatorWithPadding
from augmenter import TextAugmenter
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import nlpaug.augmenter.word as naw
import pandas as pd
from datasets import Dataset
from googletrans import Translator  # 新增回译增强
import os
from contextlib import redirect_stdout
import sys

# 1. 提前设置环境变量
os.environ["NLTK_DOWNLOAD_SILENT"] = "1" 

# 2. 静默下载依赖包
import nltk
with redirect_stdout(open(os.devnull, 'w')):  # 彻底屏蔽输出
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)  # 核心修复
    nltk.download('wordnet', quiet=True)

# 3. 继续原有代码
import logging
logging.getLogger("nltk").setLevel(logging.ERROR)  # 提升日志级别

from googletrans import Translator

# 1. 重定向所有输出到黑洞
with redirect_stdout(open(os.devnull, 'w')), redirect_stdout(sys.stderr):
    # 2. 静默下载必需NLTK包
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)  # 核心触发源[7](@ref)
    nltk.download('wordnet', quiet=True)  # SynonymAug依赖

# 3. 继续导入nlpaug
import nlpaug.augmenter.word as naw

# 使用示例（不再有NLTK提示）
aug = naw.SynonymAug(aug_src='wordnet')


# 2. 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 3. 数据预处理
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["文本"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    tokenized["labels"] = examples["类别"]  # 添加标签字段
    return tokenized

# 4. 定义评估指标
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted")
    }
    
    # 添加每个类别的F1值
    for i in range(len(model.config.id2label)):
        class_mask = (labels == i)
        if np.sum(class_mask) > 0:  # 避免除零错误
            class_f1 = f1_score(class_mask, preds == i, average="binary")
            metrics[f"f1_{model.config.id2label[i]}"] = class_f1
    return metrics

def data_loader(data_path):
    # 1. 加载数据集
    df = pd.read_csv(data_path, index_col="id")
    class_counts = df["类别"].value_counts()
    sorted_classes = class_counts.index.tolist()  # 样本量从高到低的类别名
    
    # 2. 提取类别标签并排序
    class_names = sorted(df["类别"].unique().tolist())
    
    # 3. 创建数据集并转换列类型
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("类别", ClassLabel(names=sorted_classes))
    
    # 4. 分层抽样
    dataset = ds.train_test_split(
        test_size=0.1,
        seed=42,
        stratify_by_column="类别"
    )
    
    # 5. 检查结果
    print(f"训练集: {len(dataset['train'])} 条")
    print(f"测试集: {len(dataset['test'])} 条")

    # 6. 转换数据集为DataFrame以便统计
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    
    # 7. 保存验证集
    test_df.to_csv("./dataset/val_text.csv", index=False)
    
    # 8. 创建标签映射字典
    labels_dict = {i: c for i, c in enumerate(sorted_classes)}
    print(f"类别映射: {labels_dict}")

    max_class_name_len = max(len(name) for name in class_names) + 1

    # 9. 计算并打印训练集类别分布
    train_counts = train_df["类别"].value_counts().sort_index()
    train_total = len(train_df)
    print("\n训练集类别分布:")
    for idx, count in train_counts.items():
        proportion = count / train_total * 100
        class_name = labels_dict[idx]
        print(f"  {class_name:<{max_class_name_len}} ({idx:2d}): {count:>5}条, 占比{proportion:>6.2f}%")
    
    # 10. 计算并打印测试集类别分布
    test_counts = test_df["类别"].value_counts().sort_index()
    test_total = len(test_df)
    print("\n测试集类别分布:")
    for idx, count in test_counts.items():
        proportion = count / test_total * 100
        class_name = labels_dict[idx]
        print(f"  {class_name:<{max_class_name_len}} ({idx:2d}): {count:>5}条, 占比{proportion:>6.2f}%")
    
    class_counts = []
    for cls_name in sorted_classes:
        count = df[df["类别"] == cls_name].shape[0]
        class_counts.append(count)

    # 初始化增强器（启用所有策略）
    augmenter = TextAugmenter(
        target_count=100,
        #aug_strategies=['back_trans'] 
        aug_strategies=['synonym', 'insertion']
    )
    
    # 应用增强
    balanced_df = augmenter.balance_dataset(train_df, text_col="文本", label_col="类别")
    
    # 替换原始训练集
    dataset['train'] = Dataset.from_pandas(balanced_train_df)
    
    # 打印增强后分布
    balanced_counts = balanced_train_df["类别"].value_counts().sort_index()
    print("\n增强后训练集分布:")
    for idx, count in balanced_counts.items():
        class_name = labels_dict[idx]
        print(f"  {class_name}: {count}条 (原始: {train_counts[idx]}条)")
    
    # 计算权重（保持原逻辑）
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()  # 归一化
    
    # 打印时显示百分比（更直观）
    print("\n类别权重分布:")
    max_name_len = max(len(name) for name in sorted_classes)
    for idx, (cls_name, weight) in enumerate(zip(sorted_classes, class_weights)):
        # 显示为百分比格式（如0.0667→6.67%）
        print(f"  {cls_name:<{max_name_len}} ({idx:2d}): {weight.item()*100:>6.2f}%") 

    return dataset, labels_dict, class_weights

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        :param alpha: 各类别的权重系数（Tensor），如[0.1, 0.2, ..., 0.8]
        :param gamma: 困难样本聚焦参数（越大越关注难样本）
        :param reduction: 损失聚合方式（'mean'或'sum'）
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失（每个样本单独计算）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # shape: (batch_size)
        pt = torch.exp(-ce_loss)  # 真实类别的预测概率 p_t
        
        # 应用类别权重alpha
        if self.alpha is not None:
            alpha = self.alpha[targets]  # 为每个样本选择对应类别的权重
            focal_term = alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_term = (1 - pt) ** self.gamma * ce_loss

        return focal_term.mean()

if __name__ == "__main__":

    input_path = "./dataset/train_all.csv"
    output_path= "./output/bert"
    datasets, labels_dict, class_weights = data_loader(input_path)
    tokenized_datasets = datasets.map(tokenize_function, batched=True)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels=len(labels_dict.keys()),
        id2label=labels_dict  # 标签映射
    )
    # 注入Focal Loss替换默认损失
    model.loss_fct = FocalLoss(alpha=class_weights.to(model.device),gamma=2.0)
    
    data_collator = DataCollatorWithPadding(tokenizer)

    steps = 100
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=10,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=256,
        logging_steps=steps,
        optim="adamw_torch",
        lr_scheduler_type="cosine", 
        eval_strategy="steps",
        eval_steps=steps,
        save_strategy="steps",
        logging_dir="./logs",
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=1e-6,
        max_grad_norm=1,
        fp16=True,  # GPU加速
        save_on_each_node=True,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        gradient_checkpointing=True,
        save_total_limit=1
    )
    
    # 6. 创建Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    # 7. 开始训练
    trainer.train()
    
    
    # 8. 评估模型
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        if key.startswith("f1_"):  # 高亮小类别表现
            print(f"\033[1;33m{key}: {value:.4f}\033[0m")
        else:
            print(f"{key}: {value:.4f}")
    
    # 8. 保存模型
    trainer.save_model("./my_bert_model")
    tokenizer.save_pretrained("./my_bert_model")