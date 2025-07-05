import random
import pandas as pd
from googletrans import Translator
import jieba  # 中文分词替代NLTK
import synonyms  # 轻量级同义词库
import os
import sys
import logging
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# ===== 1. 环境静默配置 =====
os.environ["GOOGLE_TRANSLATE_SILENT"] = "1"  # 抑制googletrans日志
logging.getLogger("googletrans").setLevel(logging.ERROR)

# ===== 2. 自定义增强器 =====
class TextAugmenter:
    def __init__(self, target_count=2000, aug_strategies=['synonym', 'insertion', 'back_trans']):
        self.target_count = target_count
        self.aug_strategies = aug_strategies
        self.stopwords = self._load_stopwords()  # 加载停用词表（需自定义）
        self.translator = Translator() if 'back_trans' in aug_strategies else None

    def _load_stopwords(self):
        """自定义停用词表（示例）"""
        return {"的", "了", "在", "是", "我"}  # 替换为实际停用词文件

    def _synonym_replace(self, text, replace_prob=0.3):
        """同义词替换（避免NLTK依赖）"""
        words = jieba.lcut(text)
        new_words = []
        for word in words:
            if random.random() < replace_prob and word not in self.stopwords:
                syns = synonyms.nearby(word)[0]  # 使用synonyms库获取同义词
                if syns:
                    new_words.append(random.choice(syns))
                    continue
            new_words.append(word)
        return ''.join(new_words)

    def _random_insert(self, text, insert_prob=0.3):
        """随机插入词语（自主实现）"""
        words = jieba.lcut(text)
        if len(words) < 2:
            return text
            
        new_words = words.copy()
        for _ in range(int(len(words) * insert_prob)):
            insert_idx = random.randint(0, len(new_words)-1)
            synonym = self._synonym_replace(new_words[insert_idx], replace_prob=1.0)
            new_words.insert(insert_idx, synonym)
        return ''.join(new_words)

    def back_translate(self, text, lang_path=['zh', 'en', 'fr']):
        """回译增强（保留原逻辑）"""
        try:
            translated = text
            for lang in lang_path:
                result = self.translator.translate(translated, dest=lang)
                translated = result.text
            result = self.translator.translate(translated, dest='zh-cn')
            return result.text if result.text != text else None
        except Exception:
            return None

    def augment_text(self, text, max_aug=3):
        """多策略增强入口"""
        augmented = set()
        
        if 'synonym' in self.aug_strategies:
            aug_text = self._synonym_replace(text)
            if aug_text != text:
                augmented.add(aug_text)
        
        if 'insertion' in self.aug_strategies and len(augmented) < max_aug:
            aug_text = self._random_insert(text)
            augmented.add(aug_text)
        
        if 'back_trans' in self.aug_strategies and self.translator:
            bt_text = self.back_translate(text)
            if bt_text:
                augmented.add(bt_text)
                
        return list(augmented)[:max_aug]

    def balance_dataset(self, df, text_col="文本", label_col="类别"):
        # 识别需要增强的少数类（样本量<2000）
        class_counts = df[label_col].value_counts()
        minority_classes = class_counts[class_counts < self.target_count].index.tolist()
        print(f"需增强的类别（样本量<{self.target_count}）: {minority_classes}")
        
        augmented_data = []
        for cls in minority_classes:
            minority_samples = df[df[label_col] == cls]
            current_count = len(minority_samples)
            needed = self.target_count - current_count
            print(f"类别 '{cls}': 当前{current_count}条 → 需生成{needed}条")
            
            generated_count = 0
            index = 0
            # 循环直到生成足够样本
            while generated_count < needed:
                if index >= current_count:  # 循环使用原始样本
                    index = 0
                
                sample = minority_samples.iloc[index]
                # 为每个样本生成1-3个增强版本
                new_texts = self.augment_text(sample[text_col], max_aug=3)
                
                for text in new_texts:
                    # 基础质量控制：长度和重复性
                    if len(text) > 5 and text not in df[text_col].values:
                        augmented_data.append({
                            text_col: text,
                            label_col: cls
                        })
                        generated_count += 1
                        if generated_count >= needed:
                            break
                index += 1
        
        # 合并增强数据
        return pd.concat([df, pd.DataFrame(augmented_data)], ignore_index=True)