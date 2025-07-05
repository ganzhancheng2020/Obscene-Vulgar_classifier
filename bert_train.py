# -*- coding: utf-8 -*-
# 环境安装：pip install transformers torch datasets scikit-learn
import numpy as np
import pandas as pd
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset, ClassLabel
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
import torch
from transformers import BertConfig

# 1. 初始化分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 2. 修复数据预处理函数（移至顶部避免NameError）
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["文本"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    tokenized["labels"] = examples["类别"]  # 添加标签字段
    return tokenized

# 3. 定义评估指标（增加按类别F1统计）
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

# 4. 自定义加权损失模型（提升小类别识别）
class WeightedBert(BertForSequenceClassification):
    def __init__(self, config, class_weights=None):  # 修改构造函数[9,11](@ref)
        super().__init__(config)
        self.class_weights = class_weights

    def forward(self, **inputs):
        outputs = super().forward(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        
        if labels is not None and self.class_weights is not None:
            # 应用类别权重
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device)
            )
            loss = loss_fct(
                logits.view(-1, self.num_labels), 
                labels.view(-1)
            )
            outputs.loss = loss
        return outputs

# 5. 数据加载与类别权重计算
def data_loader(data_path):
    df = pd.read_csv(data_path)
    class_names = sorted(df["类别"].unique().tolist())
    
    # 计算类别权重（提升小样本关注度）[2](@ref)
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(df["类别"]), 
        y=df["类别"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    # 创建Dataset并分层抽样
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("类别", ClassLabel(names=class_names))
    
    # 分层拆分（确保验证集分布一致）[6](@ref)
    train_idx, val_idx = train_test_split(
        range(len(ds)),
        test_size=0.1,
        random_state=42,
        stratify=ds["类别"]
    )
    train_ds = ds.select(train_idx)
    val_ds = ds.select(val_idx)
    
    # 打印分布
    print(f"训练集: {len(train_ds)} | 验证集: {len(val_ds)}")
    print(f"类别权重: {dict(zip(class_names, class_weights.numpy().round(3)))}")
    
    return {"train": train_ds, "test": val_ds}, class_weights, class_names

# 6. 主流程
if __name__ == "__main__":
    input_path = "./dataset/train_all.csv"
    output_path = "./output/bert"
    
    # 加载数据
    datasets, class_weights, class_names = data_loader(input_path)

    # 分别处理训练集和测试集
    tokenized_datasets = {
        "train": datasets["train"].map(tokenize_function, batched=True),
        "test": datasets["test"].map(tokenize_function, batched=True)
    }
    
    tokenized_datasets["train"] = tokenized_datasets["train"].map(
        lambda x: {"labels": x["类别"]}, 
        batched=True,
        remove_columns=["类别", "文本", "id"]  # 移除非输入列
    )
    tokenized_datasets["test"] = tokenized_datasets["test"].map(
        lambda x: {"labels": x["类别"]}, 
        batched=True,
        remove_columns=["类别", "文本", "id"]
    )
    print("训练集结构:", tokenized_datasets["train"])
    
    # 构建标签映射字典
    labels_dict = {i: name for i, name in enumerate(class_names)}
    
    # ==== 关键修复：正确的模型初始化方式 ====
    # 先加载配置
    config = BertConfig.from_pretrained(
        "bert-base-uncased",
        num_labels=len(class_names),
        id2label=labels_dict
    )
    # 使用配置初始化自定义模型
    model = WeightedBert(config, class_weights=class_weights)
    # 加载预训练权重
    model.load_state_dict(
        BertForSequenceClassification
        .from_pretrained("bert-base-uncased", config=config)
        .state_dict()
    )
    
    # 构建标签映射字典
    labels_dict = {i: name for i, name in enumerate(class_names)}
    data_collator = DataCollatorWithPadding(tokenizer)
    # 训练参数（关键优化点）
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=20,  # 增加轮次配合早停
        per_device_train_batch_size=32,  # 减小批次提升小样本学习
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",  # 早停依据[3](@ref)
        greater_is_better=True,
        lr_scheduler_type="cosine",
        learning_rate=2e-5,  # 优化学习率[3](@ref)
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        report_to="none",
        logging_steps=100,  # 增加日志频率
        remove_unused_columns=False
    )
    
    # 创建Trainer（注入早停和加权损失）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5
            )
        ]
    )
    
    # 训练与评估
    print("===== 开始训练 =====")
    trainer.train()
    
    print("\n===== 最终评估 =====")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        if key.startswith("f1_"):  # 高亮小类别表现
            print(f"\033[1;33m{key}: {value:.4f}\033[0m")
        else:
            print(f"{key}: {value:.4f}")
    
    # 保存最佳模型
    trainer.save_model("./my_bert_model")
    tokenizer.save_pretrained("./my_bert_model")
    print("模型已保存至 ./my_bert_model")