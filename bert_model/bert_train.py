# -*- coding: utf-8 -*-
# 环境安装：pip install transformers torch datasets scikit-learn
import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
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
import os
import random
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像

def set_seed(seed=42):
    random.seed(seed)                   # Python内置随机模块
    np.random.seed(seed)                 # NumPy随机数生成器
    torch.manual_seed(seed)             # PyTorch CPU随机种子
    torch.cuda.manual_seed(seed)        # 当前GPU随机种子
    torch.cuda.manual_seed_all(seed)     # 所有GPU随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 避免哈希随机性
    # 启用确定性计算（牺牲部分性能换取可复现性）
    torch.backends.cudnn.deterministic = True  # cuDNN确定性算法
    torch.backends.cudnn.benchmark = False     # 禁用自动优化

# 1. 初始化分词器
model_name = "/autodl-tmp"
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
def data_loader(data_path, train_path=None, test_path=None, save_path=None):
    """
    数据加载函数，支持两种模式：
    1. 从单一文件读取并自动划分训练集和验证集
    2. 直接读取已划分的训练集和验证集
    
    参数:
        data_path: 当train_path和test_path为None时，从此路径读取单一数据文件
        train_path: 训练集文件路径（可选）
        test_path: 验证集文件路径（可选）
    
    返回:
        包含训练集和验证集的字典、类别权重、类别名称
    """
    # 模式1: 直接提供训练集和验证集路径
    if train_path is not None and test_path is not None:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # 合并两个数据集以获取完整的类别列表
        combined_df = pd.concat([train_df, test_df])
        class_names = sorted(combined_df["类别"].unique().tolist())
        
        # 创建Dataset对象
        train_ds = Dataset.from_pandas(train_df)
        test_ds = Dataset.from_pandas(test_df)
        
    # 模式2: 从单一文件读取并自动划分
    else:
        df = pd.read_csv(data_path)
        class_names = sorted(df["类别"].unique().tolist())
        
        # 创建Dataset并分层抽样
        ds = Dataset.from_pandas(df)
        ds = ds.cast_column("类别", ClassLabel(names=class_names))
        
        # 分层拆分（确保验证集分布一致）
        train_idx, val_idx = train_test_split(
            range(len(ds)),
            test_size=0.2,
            random_state=42,
            stratify=ds["类别"]
        )
        train_ds = ds.select(train_idx)
        test_ds = ds.select(val_idx)

        train_df = pd.DataFrame(train_ds)
        test_df = pd.DataFrame(test_ds)
            
        train_df.to_csv(f"{save_path}train_0.8.csv", index=False)
        test_df.to_csv(f"{save_path}val_0.2.csv", index=False)
    
    # 计算类别权重（基于训练集）
    train_labels = [example["类别"] for example in train_ds]
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    
    # 确保所有数据集的类别标签一致
    for dataset in [train_ds, test_ds]:
        dataset = dataset.cast_column("类别", ClassLabel(names=class_names))
    
    # 打印分布信息
    print(f"训练集: {len(train_ds)} | 验证集: {len(test_ds)}")
    print(f"类别权重: {dict(zip(class_names, class_weights.numpy().round(3)))}")
    
    return {"train": train_ds, "test": test_ds}, class_weights, class_names

# 6. 主流程
if __name__ == "__main__":
    input_path = "./dataset/train_all_cleaned.csv"
    train_path = "./dataset/train_0.8.csv"
    test_path = "./dataset/val_0.2.csv"
    output_path = "./output/bert"
    save_path = "./dataset/"
    set_seed(42)
    
    # 加载数据
    datasets, class_weights, class_names = data_loader(
        data_path = input_path,
        train_path = train_path,
        test_path = test_path,
        save_path=save_path
    )

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
    
    # # ==== 关键修复：正确的模型初始化方式 ====
    # # 先加载配置
    # config = BertConfig.from_pretrained(
    #     model_name,
    #     num_labels=len(class_names),
    #     id2label=labels_dict
    # )
    # # 使用配置初始化自定义模型
    # model = WeightedBert(config, class_weights=class_weights)
    # # 加载预训练权重
    # model.load_state_dict(
    #     BertForSequenceClassification
    #     .from_pretrained(model_name, config=config)
    #     .state_dict()
    # )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,  # 或 "BAAI/bge-large-zh"
        num_labels=len(labels_dict.keys()),
        id2label=labels_dict
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
        save_total_limit=1,
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
        seed=42,
        report_to="swanlab",
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