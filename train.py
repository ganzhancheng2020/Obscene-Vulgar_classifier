from datasets import Dataset
from prompt import sys_prompt
import pandas as pd
import torch
from sklearn.metrics import f1_score
import numpy as n
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
# 新增早停回调类
from transformers import EarlyStoppingCallback
from peft import LoraConfig, TaskType, get_peft_model
import os
import torch
from modelscope import snapshot_download
torch.device('cuda')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_dir = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto",torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.enable_input_require_grads()
print(model.dtype)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # 过滤无效标签（-100）
    valid_indices = labels != -100
    predictions = predictions[valid_indices]
    labels = labels[valid_indices]
    print(labels)
    
    f1 = f1_score(labels, predictions, average='macro')  # 使用宏平均F1
    return {"f1": f1}

def data_processing(df):

    dataset = []
    for index, row in df.iterrows():
        text = row['文本']
        label = row['类别']
        _dict = {
            "instruction":"",
            "input":f"{sys_prompt}### 文本内容：\n{text}",
            "output":f"类别：{label}"
        }
        dataset.append(_dict)
    # 将字典列表转换为DataFrame
    new_df = pd.DataFrame(dataset)
    return new_df

def process_func(example):
    MAX_LENGTH = 512 # 设置最大序列长度为1024个token
    input_ids, attention_mask, labels = [], [], [] # 初始化返回值
    # 适配chat_template
    instruction = tokenizer(
        f"<s><|im_start|>system\n<|im_end|>\n" 
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"  
        f"<|im_start|>assistant\n",  
        add_special_tokens=False   
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    # 将instructio部分和response部分的input_ids拼接，并在末尾添加eos token作为标记结束的token
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # 注意力掩码，表示模型需要关注的位置
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    # 对于instruction，使用-100表示这些位置不计算loss（即模型不需要预测这部分）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 超出最大序列长度截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == "__main__":
    input_path = "./dataset/train_all.csv"
    df = pd.read_csv(input_path, index_col="id")
    new_df = data_processing(df)
    ds = Dataset.from_pandas(new_df)

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    dataset = tokenized_id.train_test_split(test_size=0.1)
    tokenized_id = dataset["train"]
    val_dataset = dataset["test"]
    output_dir = "./output/Qwen2.5_7B_lora"
    
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        label_names=["labels"],
        eval_strategy="steps",  # 改为按步评估
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        logging_steps=50,
        num_train_epochs=3,
        learning_rate=5e-5,
        fp16=True,
        gradient_checkpointing=True,
        load_best_model_at_end=True,  # 启用早停保存
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0
    )

    # Trainer配置
    torch.cuda.empty_cache()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer,padding=True),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.01
        )]
    )

    trainer.train()