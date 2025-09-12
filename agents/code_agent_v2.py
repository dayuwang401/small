import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from torch.nn.utils.rnn import pad_sequence
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SYSTEM_PROMPT_C = """Generate Python code to solve the problem.The code must print the result.
"""

class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, keywords, start_length):
        self.tokenizer = tokenizer
        self.start_length = start_length
        # 将传进来的字符串编译成正则对象
        self.patterns = [re.compile(p) for p in keywords]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 解码新增部分
        decoded = self.tokenizer.decode(
            input_ids[0][self.start_length:], 
            skip_special_tokens=False
        )
        
        # 逐个模式匹配
        for pat in self.patterns:
            if pat.search(decoded):
                return True
        return False

class CodeAgent_v2:
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.tokenizer, self.model = self._load_model(model_path, device, dtype)

    def _load_model(self, model_path, device, dtype):
        tok = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
        if tok.pad_token is None:
            tok.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tok))
        model.eval()
        return tok, model

    def reload_model(self):
        self.tokenizer, self.model = self._load_model(self.model_path, self.device, self.dtype)

    def build_prompt(self, reasoning_text):
        return f"{reasoning_text}\n{SYSTEM_PROMPT_C}"

    def generate_until_code_end(self, prompt, n_samples, temperature=0.7, top_p=0.95, max_new_tokens=512):
        """
        Generate text with batch parallel generation for multiple samples.
        
        Args:
            prompt: 输入提示文本
            n_samples: 生成样本数量
            temperature: 生成温度
            top_p: nucleus sampling参数
            max_new_tokens: 最大新生成token数
            
        Returns:
            tuple: (generated_texts, gen_logps, sequences)
        """
        logger.debug(f"Starting batch code generation - samples: {n_samples}, max_tokens: {max_new_tokens}")
        
        try:
            self.model.eval()
            
            # 创建batch输入：将同一个prompt复制n_samples次
            batch_prompts = [prompt] * n_samples
            logger.debug(f"Created batch with {len(batch_prompts)} prompts")
            
            # batch编码
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            input_len = inputs["input_ids"].shape[1]
            batch_size = inputs["input_ids"].shape[0]
            logger.debug(f"Batch size: {batch_size}, Input length: {input_len} tokens")
            
            with torch.no_grad():
                # 一次性batch生成所有样本
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=top_p,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                logger.debug(f"Batch generation completed successfully")
                
                # 检查生成的序列是否有效
                if outputs.sequences is None or outputs.sequences.shape[1] <= input_len:
                    logger.warning("No new tokens generated in batch")
                    # 生成默认输出
                    default_texts = ["print(0"] * n_samples
                    default_logps = torch.zeros((n_samples, 3), device=self.device)
                    default_sequences = torch.full((n_samples, input_len + 3), 
                                                 self.tokenizer.pad_token_id, 
                                                 dtype=torch.long, device=self.device)
                    return default_texts, default_logps, default_sequences
                
                # 生成部分 tokens (batch_size, generated_len)
                generated_tokens = outputs.sequences[:, input_len:]
                
                # batch解码生成的文本
                generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
                
                # 验证生成的文本
                processed_texts = []
                for i, text in enumerate(generated_texts):
                    if not text or not text.strip():
                        logger.warning(f"Sample {i+1}: Empty generated text, using default")
                        processed_texts.append("print(0)")
                    else:
                        processed_texts.append(text)
                
                logger.debug(f"Generated {len(processed_texts)} text samples")
                
                # 计算 log probs for batch
                gen_logps = self._compute_generation_logps_batch(outputs, generated_tokens)
                
                logger.debug(f"Final shapes - texts: {len(processed_texts)}, "
                            f"logps: {gen_logps.shape}, sequences: {outputs.sequences.shape}")
                
                # 清理极端值
                gen_logps = self._clean_extreme_values(gen_logps, "generation log probabilities")
                
                logger.debug("Batch code generation completed successfully")
                return processed_texts, gen_logps, outputs.sequences
                
        except Exception as e:
            logger.error(f"Critical error in batch generate_until_code_end: {e}")
            
            # 返回安全的默认值
            default_texts = ["print(0"] * n_samples
            default_logps = torch.zeros((n_samples, 3), device=self.device)
            default_sequences = torch.full((n_samples, input_len + 3 if 'input_len' in locals() else 10), 
                                         self.tokenizer.pad_token_id, 
                                         dtype=torch.long, device=self.device)
            
            return default_texts, default_logps, default_sequences
    
    def _compute_generation_logps_batch(self, outputs, generated_tokens, clamp_min=-10.0, clamp_max=0.0):
        """
        安全地计算batch生成token的log概率
        
        Args:
            outputs: 模型生成的输出
            generated_tokens: 生成的token序列 (batch_size, seq_len)
            clamp_min: log概率最小值
            clamp_max: log概率最大值
            
        Returns:
            torch.Tensor: 处理后的log概率 (batch_size, seq_len)
        """
        try:
            batch_size, seq_len = generated_tokens.shape
            
            if hasattr(outputs, 'scores') and outputs.scores:
                logger.debug(f"Computing batch log probabilities for {len(outputs.scores)} tokens, batch_size: {batch_size}")
                
                # 收集所有位置的log概率
                all_logps = []
                
                for i, scores in enumerate(outputs.scores):
                    if i < seq_len:
                        # 预处理scores，防止极端值
                        scores = torch.clamp(scores.float(), min=-50.0, max=50.0)
                        
                        # 计算log_softmax (batch_size, vocab_size)
                        log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
                        
                        # 安全地提取对应token的log概率
                        token_indices = generated_tokens[:, i].unsqueeze(-1)  # (batch_size, 1)
                        
                        # 确保索引在有效范围内
                        token_indices = torch.clamp(token_indices, 0, log_probs.shape[-1] - 1)
                        
                        # 提取每个样本对应token的log概率
                        selected_logps = torch.gather(log_probs, dim=-1, index=token_indices).squeeze(-1)  # (batch_size,)
                        
                        # 裁剪到合理范围
                        selected_logps = torch.clamp(selected_logps, min=clamp_min, max=clamp_max)
                        
                        all_logps.append(selected_logps)
                
                if all_logps:
                    # 堆叠成 (batch_size, seq_len)
                    gen_logps = torch.stack(all_logps, dim=1)
                    logger.debug(f"Batch log probabilities computed successfully, shape: {gen_logps.shape}")
                else:
                    logger.warning("No valid log probabilities computed from scores")
                    gen_logps = torch.full((batch_size, 1), clamp_min, device=self.device)
            else:
                logger.warning("No scores available in generation output")
                gen_logps = torch.full((batch_size, seq_len), clamp_min, device=self.device)
            
            # 最终安全检查
            gen_logps = self._clean_extreme_values(gen_logps, "batch token log probabilities")
            
            return gen_logps
            
        except Exception as e:
            logger.error(f"Error computing batch generation log probabilities: {e}")
            # 返回安全的默认值
            batch_size = generated_tokens.shape[0] if generated_tokens is not None else 1
            seq_len = generated_tokens.shape[1] if generated_tokens is not None and len(generated_tokens.shape) > 1 else 1
            return torch.full((batch_size, seq_len), clamp_min, device=self.device)
    
    def _clean_extreme_values(self, tensor, tensor_name, clamp_min=-10.0, clamp_max=10.0):
        """
        清理张量中的极端值
        
        Args:
            tensor: 要清理的张量
            tensor_name: 张量名称，用于日志
            clamp_min: 最小值
            clamp_max: 最大值
            
        Returns:
            torch.Tensor: 清理后的张量
        """
        try:
            original_shape = tensor.shape
            
            # 检查并处理NaN值
            if torch.isnan(tensor).any():
                logger.warning(f"NaN values detected in {tensor_name}, replacing with {clamp_min}")
                tensor = torch.nan_to_num(tensor, nan=clamp_min)
            
            # 检查并处理无穷值
            if torch.isinf(tensor).any():
                logger.warning(f"Inf values detected in {tensor_name}, replacing with extreme values")
                tensor = torch.nan_to_num(tensor, neginf=clamp_min, posinf=clamp_max)
            
            # 裁剪到合理范围
            tensor = torch.clamp(tensor, min=clamp_min, max=clamp_max)
            
            # 验证形状没有改变
            if tensor.shape != original_shape:
                logger.error(f"Shape changed during cleaning: {original_shape} -> {tensor.shape}")
            
            # 最终验证
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logger.error(f"Failed to clean extreme values in {tensor_name}")
                # 创建安全的替代张量
                tensor = torch.full_like(tensor, clamp_min)
            
            return tensor
            
        except Exception as e:
            logger.error(f"Error cleaning extreme values in {tensor_name}: {e}")
            # 返回安全的默认张量
            return torch.full_like(tensor, clamp_min) if tensor is not None else torch.tensor([clamp_min])
