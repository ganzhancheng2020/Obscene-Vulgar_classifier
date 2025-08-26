#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

def load_model(model_path="./my_bert_model"):
    """加载预训练模型和分词器"""
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, tokenizer, device

def predict_batch(texts, model, tokenizer, device, batch_size=32, max_length=256):
    """批量预测文本列表"""
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="预测进度"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_indices = torch.argmax(probs, dim=1).cpu().numpy()
        
        for idx in pred_indices:
            # 直接获取标签名称，不保留置信度和ID
            results.append(model.config.id2label[idx])
    return results

if __name__ == "__main__":
    # 参数解析（默认路径设置）
    parser = argparse.ArgumentParser(description="BERT文本分类批量预测")
    parser.add_argument("--input", type=str, default="./dataset/test_text.csv", 
                        help="输入文件路径（默认：dataset/test_text.csv）")
    parser.add_argument("--output", type=str, default="./dataset/submit.csv", 
                        help="输出文件路径（默认：submit.csv）")
    parser.add_argument("--model", type=str, default="./my_bert_model", 
                        help="模型目录路径（默认：./my_bert_model）")
    args = parser.parse_args()

    # 1. 加载模型
    model, tokenizer, device = load_model(args.model)
    print(f"✅ 模型加载完成 | 设备: {device}")
    print("🔍 标签映射:", model.config.id2label)

    # 2. 读取数据（处理三列格式）
    try:
        # 读取包含三列（ID、类别、文本）的CSV文件
        df = pd.read_csv(args.input)
        
        # 验证必须包含的列
        required_columns = ["id", "文本"]
        if not set(required_columns).issubset(df.columns):
            missing = set(required_columns) - set(df.columns)
            raise KeyError(f"CSV文件缺少必要列: {missing}")
        
        texts = df["文本"].tolist()
        ids = df["id"].tolist()
        print(f"📥 加载数据: {args.input} | 样本数: {len(texts)}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        exit(1)

    # 3. 批量预测
    predicted_labels = predict_batch(texts, model, tokenizer, device)

    # 4. 保存结果（仅保留ID和文本类别两列）
    result_df = pd.DataFrame({
        "id": ids,
        "类别": predicted_labels
    })
    result_df.to_csv(args.output, index=False)
    
    # 5. 统计报告
    label_counts = result_df["类别"].value_counts()
    print(f"\n✅ 预测完成! 结果保存至: {args.output}")
    print("📊 类别分布统计:")
    print(label_counts.to_string())