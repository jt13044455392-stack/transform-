from typing import Dict, Tuple
from transformers import AutoTokenizer, AutoModel

class AttentionModelLoader:
    """通用的模型加载器"""
    
    def __init__(self):
        # 支持的模型配置
        self.model_configs = {
            "BERT Base": {
                "name": "bert-base-uncased",
                "type": "encoder",
                "max_length": 512,
                "description": "谷歌的BERT基础模型",
                "intro": "BERT Base 是双向编码器模型，适合理解句子含义、分类、问答和词语关系分析。",
                "features": "12层 Transformer，12个注意力头，约1.1亿参数。",
                "best_for": "文本理解、注意力机制入门观察。"
            },
            "BERT Chinese": {
                "name": "bert-base-chinese",
                "type": "encoder",
                "max_length": 512,
                "description": "谷歌发布的中文BERT基础模型",
                "intro": "BERT Chinese 是面向中文文本的 BERT 模型，适合观察中文词字级别的上下文关系。",
                "features": "12层 Transformer，12个注意力头，约1.1亿参数。",
                "best_for": "中文文本理解、中文注意力可视化。"
            },
            "Chinese BERT WWM": {
                "name": "hfl/chinese-bert-wwm-ext",
                "type": "encoder",
                "max_length": 512,
                "description": "哈工大讯飞联合实验室发布的中文全词掩码BERT",
                "intro": "Chinese BERT WWM 使用全词掩码训练方式，更适合处理中文词语整体语义。",
                "features": "12层 Transformer，12个注意力头，基于中文全词掩码预训练。",
                "best_for": "中文词语级语义分析、和普通中文 BERT 对比。"
            },
            "GPT-2": {
                "name": "gpt2",
                "type": "decoder",
                "max_length": 1024,
                "description": "OpenAI的GPT-2模型",
                "intro": "GPT-2 是自回归解码器模型，按从左到右的顺序处理文本，常用于文本生成。",
                "features": "12层 Transformer，12个注意力头，约1.17亿参数。",
                "best_for": "观察生成式模型如何关注前文。"
            },
            "RoBERTa": {
                "name": "roberta-base",
                "type": "encoder",
                "max_length": 512,
                "description": "Facebook改进的BERT",
                "intro": "RoBERTa 改进了 BERT 的训练方式，通常在文本理解任务上表现更稳定。",
                "features": "12层 Transformer，12个注意力头，约1.25亿参数。",
                "best_for": "和 BERT 对比编码器注意力模式。"
            },
            "DistilBERT": {
                "name": "distilbert-base-uncased",
                "type": "encoder",
                "max_length": 512,
                "description": "蒸馏版BERT，速度快3倍",
                "intro": "DistilBERT 是压缩版 BERT，保留主要理解能力，同时减少模型大小和计算量。",
                "features": "6层 Transformer，12个注意力头，约6600万参数。",
                "best_for": "快速运行和轻量级注意力可视化。"
            }
        }
        
        self.loaded_models = {}  # 缓存加载的模型

    def validate_model_key(self, model_key: str) -> None:
        """检查模型key是否存在"""
        if model_key not in self.model_configs:
            available_models = ", ".join(self.get_available_models())
            raise ValueError(f"不支持的模型: {model_key}。可用模型: {available_models}")
    
    def get_available_models(self) -> list:
        """获取可用模型列表"""
        return list(self.model_configs.keys())
    
    def load(self, model_key: str) -> Tuple[AutoTokenizer, AutoModel]:
        """加载模型和tokenizer"""
        self.validate_model_key(model_key)
        
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        config = self.model_configs[model_key]
        model_name = config["name"]
        
        print(f"正在加载 {model_key}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if config["type"] == "decoder" and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModel.from_pretrained(model_name, output_attentions=True)
        model.eval()
        
        # 缓存
        self.loaded_models[model_key] = (tokenizer, model)
        
        return tokenizer, model
    
    def get_model_info(self, model_key: str) -> Dict:
        """获取模型信息"""
        self.validate_model_key(model_key)
        config = self.model_configs[model_key]
        return {
            "name": config["name"],
            "type": config["type"],
            "max_length": config["max_length"],
            "description": config["description"],
            "intro": config["intro"],
            "features": config["features"],
            "best_for": config["best_for"]
        }

class UniversalTokenizer:
    """统一处理不同模型的tokenizer"""
    
    def __init__(self, tokenizer, model_type="encoder"):
        self.tokenizer = tokenizer
        self.model_type = model_type
    
    def encode(self, text: str, max_length: int = 512):
        """统一的编码方法"""
        
        # 不同模型处理方式不同
        if self.model_type == "encoder":
            # BERT类：自动添加[CLS]和[SEP]
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
        else:  # decoder (GPT-2)
            # GPT-2：从左往右生成
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
        
        return inputs
    
    def get_tokens(self, input_ids):
        """统一获取token列表"""
        # 移除特殊token
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # 清理：移除[CLS], [SEP]等
        cleaned_tokens = []
        for token in tokens:
            if token.startswith('[') and token.endswith(']'):
                continue  # 跳过BERT的特殊token
            if token == '<|endoftext|>':
                continue  # 跳过GPT的特殊token
            cleaned_tokens.append(token)
        
        return cleaned_tokens
