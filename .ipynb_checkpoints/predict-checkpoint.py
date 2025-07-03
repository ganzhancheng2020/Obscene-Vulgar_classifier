from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompt import sys_prompt, sys_prompt2
import os
import json
import pandas as pd  # 新增pandas库
from modelscope import snapshot_download

import re

def filter_reasoning(text: str) -> str:
    # 匹配并删除 <think>...</think> 及其内部内容
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 去除多余空行
    return re.sub(r'\n\s*\n', '\n', cleaned).strip()
    
# 自动下载模型时，指定使用modelscope; 否则，会从HuggingFace下载
os.environ['VLLM_USE_MODELSCOPE']='True'
model_dir = snapshot_download('Qwen/Qwen3-8B', cache_dir='/root/autodl-tmp', revision='master')

def get_completion(prompts, model, tokenizer=None, max_tokens=2048, **kwargs):  # 缩短max_tokens
    stop_token_ids = [151643, 151645]  # 修正停止符顺序
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids
    )
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len,trust_remote_code=True)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def to_submit(df, output_path):

    df = df[['id', '类别']]
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存至 {output_path}")
    


if __name__ == "__main__":
    # 初始化 vLLM 推理引擎
    model='/root/autodl-tmp/Qwen/Qwen3-8B' # 指定模型路径
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False) # 加载分词器
    max_model_len = 4096 # 模型最大长度
    max_token = 100 #模型输出最大长度
    text_col = '文本' # 文本列名
    label_col = '类别' # 标签列名
    thinking = False # 是否开启推理模式
    
    # 新增CSV读取处理
    input_path = "./dataset/test_text.csv"
    output_path = "./dataset/submit.csv"
    
    try:
        df = pd.read_csv(input_path)
        if text_col not in df.columns:
            raise ValueError("CSV文件中缺少'文本'列")
            
        print(f"正在处理 {len(df)} 条数据...")
        
        # 批量生成提示
        prompts = []
        for text in df['文本']:
            messages = [
                {"role": "user", "content": f"{sys_prompt}### 文本内容：\n{text}"}
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking
            ))
        print(messages)
        # 批量推理
        outputs = get_completion(prompts, model, max_tokens=max_token)
        
        # 提取结果并保存
        if thinking: 
            df[label_col] = [filter_reasoning(output.outputs[0].text.split("：")[-1]) for output in outputs]
        else:
            df[label_col] = [output.outputs[0].text.split("：")[-1].strip() for output in outputs]

        to_submit(df, output_path)
        
    except Exception as e:
        print(f"处理失败: {str(e)}")