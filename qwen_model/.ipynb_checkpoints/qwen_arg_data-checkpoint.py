from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json
import re
from modelscope import snapshot_download
import pandas as pd

# 自动下载模型时，指定使用modelscope; 否则，会从HuggingFace下载
os.environ['VLLM_USE_MODELSCOPE'] = 'True'

# 初始化模型和分词器
model_dir = snapshot_download('Qwen/Qwen3-8B', cache_dir='/root/autodl-tmp', revision='master')
model = '/root/autodl-tmp/Qwen/Qwen3-8B'  # 指定模型路径
tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)  # 加载分词器

# 初始化 vLLM 推理引擎（全局一次初始化）
stop_token_ids = [151645, 151643]
sampling_params = SamplingParams(
    temperature=1.0, 
    top_p=0.8, 
    top_k=20, 
    min_p=0, 
    max_tokens=4096, 
    stop_token_ids=stop_token_ids
)
llm = LLM(
    model=model, 
    max_model_len=8192,
    trust_remote_code=True
)

def batch_data_augmentation(batch_types, batch_inputs):
    """
    批量处理数据增强
    """
    prompts = []
    for types, inputs in zip(batch_types, batch_inputs):
        prompt = f"""你是一名专注于网络内容安全的数据增强专家。你的唯一任务是对用户输入的**违规文本**进行**同义改写**，用于训练内容审核模型。

请严格遵守以下核心法则：
1. **保留违规核心（最重要）**：改写必须**绝对保持原句的违规性质、恶意意图和严重程度**。不能将其弱化为非违规内容，也不能加剧其严重性导致失真。
2. **关键信息锁定**：原句中的以下关键元素**必须完全保留，不允许任何更改**：
    *   **具体敏感词**（如脏话、侮辱性称谓、违禁品名称）
    *   **实体信息**（如人名、地名、组织名、网址、联系方式）
    *   **数字信息**（如价格、时间、数量）
    *   **违规具体手法**（如诈骗步骤、违禁品交易方式）
3. **多样化技巧**：在遵守上述两条的前提下，你可以对句子的**其他部分**进行灵活改写，技巧包括：
    *   同义词/近义词替换（**非关键词**，如动词、形容词、副词）
    *   句式结构调整（主动变被动、调整语序）
    *   语气变化（如将陈述句改为反问句）
    *   添加/删除无关紧要的修饰词（如"真的"、"非常"）
4. **自然性与真实性**：改写后的文本必须符合网络语境（如社交媒体、论坛、聊天室）的真实表达习惯，听起来像一个真实用户会说出的话，避免生硬和不自然。
    
**绝对禁止项（Negative Prompting）**：
*   **禁止**改变原句的违规类别（如不能把仇恨言论改写成广告 spam）。
*   **禁止**净化、美化或模糊任何敏感词和恶意意图。
*   **禁止**添加任何会改变原句核心含义的额外信息。
*   **禁止**添加任何序号。
    
**输出格式**：
对于输入的每个句子，直接生成5种不同的改写版本。每个版本单独一行，并标上序号。无需任何解释。
    
**用户输入的句子为{types}类别，请对以下违规句子进行安全改写：**
{inputs}
"""
        
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        prompts.append(text)
    
    # 批量生成
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def augment_train_data(csv_path, output_path=None, sample_size=None, batch_size=8, 
                      minority_threshold=0.3, target_balance_ratio=0.8):
    """
    读取CSV文件并批量调用数据增强函数，只对少数类别进行增强
    
    参数:
        csv_path: 输入的CSV文件路径
        output_path: 增强后数据的保存路径（可选）
        sample_size: 采样的数据量（可选，用于测试）
        batch_size: 批量处理的大小
        minority_threshold: 少数类别的阈值（相对于最多类别的比例）
        target_balance_ratio: 目标平衡比例，增强后少数类别的样本数量达到最多类别的比例
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 如果指定了采样大小，则进行采样
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    # 分析类别分布
    class_counts = df['类别'].value_counts()
    max_count = class_counts.max()
    print("原始数据类别分布:")
    print(class_counts)
    
    # 确定需要增强的少数类别（样本数少于最大类别的一定比例）
    minority_classes = class_counts[class_counts < max_count * minority_threshold].index.tolist()
    print(f"需要增强的少数类别: {minority_classes}")
    
    # 如果没有少数类别，直接返回原始数据
    if len(minority_classes) == 0:
        print("没有需要增强的少数类别，直接返回原始数据")
        if output_path:
            df.to_csv(output_path, index=False, encoding='utf-8')
        return df
    
    # 计算每个少数类别需要增强的数量
    augmentation_plan = {}
    for cls in minority_classes:
        current_count = class_counts[cls]
        target_count = int(max_count * target_balance_ratio)
        needed_augment = max(0, target_count - current_count)
        augmentation_plan[cls] = {
            'current_count': current_count,
            'target_count': target_count,
            'needed_augment': needed_augment,
            'augment_per_sample': min(5, max(1, needed_augment // current_count)) if current_count > 0 else 5
        }
    
    print("增强计划:")
    for cls, plan in augmentation_plan.items():
        print(f"类别 {cls}: 当前 {plan['current_count']}, 目标 {plan['target_count']}, " +
              f"需要增强 {plan['needed_augment']}, 每个样本增强 {plan['augment_per_sample']} 个")
    
    # 提取少数类别的数据
    minority_df = df[df['类别'].isin(minority_classes)].copy()
    
    # 初始化结果列表
    augmented_data = []
    
    # 分批处理少数类别数据
    for cls in minority_classes:
        cls_df = minority_df[minority_df['类别'] == cls]
        plan = augmentation_plan[cls]
        
        print(f"\n处理类别 {cls}: 需要从 {len(cls_df)} 个样本中生成 {plan['needed_augment']} 个增强样本")
        
        # 如果不需要增强，跳过
        if plan['needed_augment'] <= 0:
            continue
            
        # 计算需要处理的样本数量
        samples_needed = min(len(cls_df), plan['needed_augment'] // plan['augment_per_sample'] + 1)
        cls_df_to_process = cls_df.iloc[:samples_needed]
        
        for i in range(0, len(cls_df_to_process), batch_size):
            batch_df = cls_df_to_process.iloc[i:i+batch_size]
            batch_texts = batch_df['文本'].tolist()
            batch_labels = batch_df['类别'].tolist()
            batch_ids = batch_df['id'].tolist()
            
            print(f"处理批次 {i//batch_size + 1}/{(len(cls_df_to_process)-1)//batch_size + 1}: " +
                  f"样本 {i+1}-{min(i+batch_size, len(cls_df_to_process))}")
            
            try:
                # 批量调用数据增强函数
                outputs = batch_data_augmentation(batch_labels, batch_texts)
                
                # 解析每个输出
                for j, output in enumerate(outputs):
                    original_idx = i + j
                    if original_idx >= len(batch_df):
                        break
                        
                    original_text = batch_df.iloc[original_idx]['文本']
                    label = batch_df.iloc[original_idx]['类别']
                    original_id = batch_df.iloc[original_idx]['id']
                    
                    generated_text = output.outputs[0].text
                    
                    # 清理生成的文本
                    lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
                    
                    # 移除行首的序号
                    cleaned_lines = []
                    for line in lines:
                        cleaned_line = re.sub(r'^\d+[\.\、]?\s*', '', line)
                        cleaned_lines.append(cleaned_line)
                    
                    # 将每个增强后的文本添加到结果中，并为每个增强文本生成新的ID
                    for idx, aug_text in enumerate(cleaned_lines[:plan['augment_per_sample']]):
                        if aug_text and len(aug_text) > 3:
                            # 生成新的ID：原始ID + "_aug" + 序号
                            new_id = f"{original_id}_aug{idx+1}"
                            
                            augmented_data.append({
                                'id': new_id,  # 使用新的ID
                                '文本': aug_text,
                                '类别': label
                            })
                
                # 检查是否已达到目标增强数量
                current_augmented = len([d for d in augmented_data if d['类别'] == cls])
                if current_augmented >= plan['needed_augment']:
                    print(f"类别 {cls} 已达到目标增强数量 {plan['needed_augment']}")
                    break
                    
            except Exception as e:
                print(f"处理批次 {i//batch_size + 1} 时出错: {e}")
                # 记录出错的具体样本
                for j in range(len(batch_texts)):
                    original_idx = i + j
                    if original_idx < len(df):
                        print(f"出错样本: {df.iloc[original_idx]['文本']}")
                continue
                
            except Exception as e:
                print(f"处理批次 {i//batch_size + 1} 时出错: {e}")
                # 记录出错的具体样本
                for j in range(len(batch_texts)):
                    original_idx = i + j
                    if original_idx < len(df):
                        print(f"出错样本: {df.iloc[original_idx]['文本']}")
                continue
                
    # 创建增强后的DataFrame
    augmented_df = pd.DataFrame(augmented_data)
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    
    # 如果指定了输出路径，则保存结果
    if output_path:
        combined_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"增强后的数据已保存至: {output_path}")
    
    # 打印统计信息
    print(f"原始数据量: {len(df)}")
    print(f"增强后数据量: {len(augmented_df)}")
    print("类别分布:")
    print(augmented_df['类别'].value_counts())
    
    return augmented_df

# 使用示例
if __name__ == "__main__":
    # 调用函数进行数据增强
    augmented_df = augment_train_data(
        csv_path="/root/Obscene-Vulgar_classifier/bert_model/dataset/train_all.csv",
        output_path="/root/Obscene-Vulgar_classifier/bert_model/dataset/train_all_aug.csv",
        batch_size=4096     # 批量大小，根据GPU内存调整
    )