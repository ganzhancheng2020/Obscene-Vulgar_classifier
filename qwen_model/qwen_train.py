from datasets import Dataset
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from peft import LoraConfig, TaskType, get_peft_model  # 添加 TaskType
from transformers import (
    Qwen2ForSequenceClassification,  # 关键修改：使用序列分类模型
    Qwen2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,  # 使用适合分类任务的DataCollator
    EarlyStoppingCallback,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from peft import LoraConfig, get_peft_model
import os
from tqdm import tqdm
from modelscope import snapshot_download
from sklearn.preprocessing import LabelEncoder

# 设置环境
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型和分词器 (Qwen2专用)
model_dir = "Qwen/Qwen3-8B"  # 替换为实际模型路径
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# 数据处理函数 (适配分类任务)
def process_func(examples):
    """
    处理文本数据为模型输入格式
    返回: input_ids, attention_mask, labels
    """
    
    # 分词处理
    tokenized = tokenizer(
        text=examples["文本"],
        padding="max_length",
        truncation=True,
        max_length=256,  # 根据显存调整[3](@ref)
        return_tensors="pt"
    )
    
    # 添加标签
    tokenized["labels"] = torch.tensor(examples["label_idx"], dtype=torch.long)
    return tokenized

# 评估指标计算
def compute_metrics(eval_pred):
    """
    计算分类任务的评估指标
    返回: dict(f1, accuracy)
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        "f1": f1_score(labels, predictions, average="macro"),
        "accuracy": accuracy_score(labels, predictions)
    }

# 数据集准备
def prepare_dataset(file_path):
    """
    准备训练/验证数据集
    返回: Dataset(tokenized)
    """
    df = pd.read_csv(file_path)
    
    # 构建数据集 (假设列名: id, text, label)
    dataset = Dataset.from_pandas(df[["文本", "类别"]])
    
    # 应用预处理
    return dataset.map(process_func, batched=True, remove_columns=["文本"])

if __name__ == "__main__":
    # 准备数据
    input_path = "./dataset/train_all.csv"
    df = pd.read_csv(input_path, index_col="id")
    label_encoder = LabelEncoder()
    df['label_idx'] = label_encoder.fit_transform(df['类别'])  # 将类别名称转换为数字索引

    ds = Dataset.from_pandas(df)

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    dataset = tokenized_id.train_test_split(test_size=0.1)

    output_dir = "./output/Qwen3-8B_lora"
    trained_model_path = "/root/data/Qwen/merged_qwen3_8b_lora"

    label_encoder = LabelEncoder()
    df['label_idx'] = label_encoder.fit_transform(df['类别'])
    # 获取类别名称列表（按编码顺序）
    class_names = label_encoder.classes_.tolist()
    # 构建 id2label 映射
    id2label = {i: name for i, name in enumerate(class_names)}

    # 加载序列分类模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=10,  # 根据实际类别数修改
        device_map="auto",
        torch_dtype=torch.bfloat16,
        id2label=id2label
    )
    model.enable_input_require_grads()
    model.config.use_cache = False
    print(model)

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
    )
    model = get_peft_model(model, config)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        eval_strategy="steps",  # 改为按步评估
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=50,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        load_best_model_at_end=True,  # 启用早停保存
        metric_for_best_model="loss",
        max_grad_norm=1.0,
        save_total_limit=1
    )

    # Trainer配置
    torch.cuda.empty_cache()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),  # 分类专用collator
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    # 开始训练
    trainer.train()
    
    # 保存最佳模型
    trainer.save_model("./output/Qwen3-classification-best")