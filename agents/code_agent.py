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

class CodeAgent:
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
        Generate text with keyword stopping for a single prompt, multiple samples (variable length OK),
        and return generated_texts, gen_logps, outputs.sequences (padded to max length).
        
        Args:
            prompt: 输入提示文本
            n_samples: 生成样本数量
            temperature: 生成温度
            top_p: nucleus sampling参数
            max_new_tokens: 最大新生成token数
            
        Returns:
            tuple: (generated_texts, gen_logps, sequences)
        """
        logger.debug(f"Starting code generation - samples: {n_samples}, max_tokens: {max_new_tokens}")
        
        try:
            self.model.eval()
            
            # 单条输入进行编码
            logger.debug("Encoding input prompt")
            inputs = self.tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            input_len = inputs["input_ids"].shape[1]
            logger.debug(f"Input length: {input_len} tokens")
            
            # 停止条件：遇到 "</code>"
            stopping_criteria = StoppingCriteriaList([
                                            KeywordStoppingCriteria(self.tokenizer, keywords=[r"```python\s*([\s\S]*?)```"], start_length=input_len)
                                        ])
            
            all_generated_texts = []
            all_gen_logps_list = []
            all_sequences_list = []
            
            logger.debug("Starting generation loop")
            
            with torch.no_grad():
                for sample_idx in range(n_samples):
                    logger.debug(f"Generating sample {sample_idx + 1}/{n_samples}")
                    
                    try:
                        # 生成文本
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            top_k=50,
                            top_p=top_p,
                            temperature=temperature,
                            stopping_criteria=stopping_criteria,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            return_dict_in_generate=True,
                            output_scores=True
                        )
                        
                        logger.debug(f"Sample {sample_idx + 1} generated successfully")
                        
                        # 检查生成的序列是否有效
                        if outputs.sequences is None or outputs.sequences.shape[1] <= input_len:
                            logger.warning(f"Sample {sample_idx + 1}: No new tokens generated")
                            # 生成默认输出
                            default_text = "print(0)"
                            all_generated_texts.append(default_text)
                            all_gen_logps_list.append(torch.zeros(len(default_text.split()), device=self.device))
                            
                            # 创建默认序列
                            default_tokens = self.tokenizer.encode(default_text, return_tensors="pt").to(self.device)
                            default_sequence = torch.cat([inputs["input_ids"], default_tokens], dim=1).squeeze(0)
                            all_sequences_list.append(default_sequence)
                            continue
                        
                        # 生成部分 tokens
                        generated_tokens = outputs.sequences[:, input_len:]
                        generated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
                        
                        # 验证生成的文本
                        if not generated_text or not generated_text[0].strip():
                            logger.warning(f"Sample {sample_idx + 1}: Empty generated text")
                            generated_text = ["print(0)"]
                        
                        all_generated_texts.extend(generated_text)
                        logger.debug(f"Sample {sample_idx + 1} text length: {len(generated_text[0])}")
                        
                        # 计算 log probs with numerical stability
                        gen_logps = self._compute_generation_logps(outputs, generated_tokens)
                        
                        all_gen_logps_list.append(gen_logps.squeeze(0))   # 去掉 batch 维度
                        all_sequences_list.append(outputs.sequences.squeeze(0))  # (seq_total_len,)
                        
                        # 清理当前样本的输出
                        del outputs
                        
                    except Exception as e:
                        logger.error(f"Error generating sample {sample_idx + 1}: {e}")
                        
                        # 生成默认输出作为fallback
                        default_text = "print(0)"
                        all_generated_texts.append(default_text)
                        all_gen_logps_list.append(torch.zeros(3, device=self.device))  # 假设3个token
                        
                        # 创建默认序列
                        default_tokens = self.tokenizer.encode(default_text, return_tensors="pt").to(self.device)
                        default_sequence = torch.cat([inputs["input_ids"], default_tokens], dim=1).squeeze(0)
                        all_sequences_list.append(default_sequence)
            
            logger.debug("Generation completed, processing outputs")
            
            # 验证输出数量
            if len(all_generated_texts) != n_samples:
                logger.error(f"Generated {len(all_generated_texts)} texts but expected {n_samples}")
                # 补齐缺失的样本
                while len(all_generated_texts) < n_samples:
                    all_generated_texts.append("print(0)")
                    all_gen_logps_list.append(torch.zeros(3, device=self.device))
                    default_tokens = self.tokenizer.encode("print(", return_tensors="pt").to(self.device)
                    default_sequence = torch.cat([inputs["input_ids"], default_tokens], dim=1).squeeze(0)
                    all_sequences_list.append(default_sequence)
            
            # 对不同长度进行 padding with safety checks
            logger.debug("Padding sequences to same length")
            
            if not all_gen_logps_list:
                logger.error("No valid log probabilities generated")
                all_gen_logps_list = [torch.zeros(1, device=self.device) for _ in range(n_samples)]
            
            if not all_sequences_list:
                logger.error("No valid sequences generated")
                all_sequences_list = [torch.zeros(input_len + 1, dtype=torch.long, device=self.device) 
                                    for _ in range(n_samples)]
            
            # 安全的padding操作
            try:
                all_gen_logps = pad_sequence(all_gen_logps_list, batch_first=True, padding_value=0.0)
                all_sequences = pad_sequence(all_sequences_list, batch_first=True, 
                                           padding_value=self.tokenizer.pad_token_id)
                
                logger.debug(f"Final shapes - texts: {len(all_generated_texts)}, "
                            f"logps: {all_gen_logps.shape}, sequences: {all_sequences.shape}")
                
            except Exception as e:
                logger.error(f"Error during padding: {e}")
                # 创建默认输出
                all_gen_logps = torch.zeros((n_samples, max_new_tokens), device=self.device)
                all_sequences = torch.full((n_samples, input_len + max_new_tokens), 
                                         self.tokenizer.pad_token_id, 
                                         dtype=torch.long, device=self.device)
            
            # 最终验证和清理
            all_gen_logps = self._clean_extreme_values(all_gen_logps, "generation log probabilities")
            
            logger.debug("Code generation completed successfully")
            return all_generated_texts, all_gen_logps, all_sequences
            
        except Exception as e:
            logger.error(f"Critical error in generate_until_code_end: {e}")
            
            # 返回安全的默认值
            default_texts = ["print(0)"] * n_samples
            default_logps = torch.zeros((n_samples, 3), device=self.device)
            default_sequences = torch.full((n_samples, input_len + 3), 
                                         self.tokenizer.pad_token_id, 
                                         dtype=torch.long, device=self.device)
            
            return default_texts, default_logps, default_sequences
    
    def _compute_generation_logps(self, outputs, generated_tokens, clamp_min=-10.0, clamp_max=0.0):
        """
        安全地计算生成token的log概率
        
        Args:
            outputs: 模型生成的输出
            generated_tokens: 生成的token序列
            clamp_min: log概率最小值
            clamp_max: log概率最大值
            
        Returns:
            torch.Tensor: 处理后的log概率
        """
        try:
            gen_logps = []
            
            if hasattr(outputs, 'scores') and outputs.scores:
                logger.debug(f"Computing log probabilities for {len(outputs.scores)} tokens")
                
                for i, scores in enumerate(outputs.scores):
                    if i < generated_tokens.shape[1]:
                        # 预处理scores，防止极端值
                        scores = torch.clamp(scores.float(), min=-50.0, max=50.0)
                        
                        # 计算log_softmax
                        log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
                        
                        # 安全地提取对应token的log概率
                        token_idx = generated_tokens[:, i].unsqueeze(-1)
                        
                        # 确保索引在有效范围内
                        token_idx = torch.clamp(token_idx, 0, log_probs.shape[-1] - 1)
                        
                        selected_logps = torch.gather(log_probs, dim=-1, index=token_idx).squeeze(-1)
                        
                        # 裁剪到合理范围
                        selected_logps = torch.clamp(selected_logps, min=clamp_min, max=clamp_max)
                        
                        gen_logps.append(selected_logps)
                
                if gen_logps:
                    gen_logps = torch.stack(gen_logps, dim=1)  # (1, seq_len)
                    logger.debug(f"Log probabilities computed successfully, shape: {gen_logps.shape}")
                else:
                    logger.warning("No valid log probabilities computed from scores")
                    gen_logps = torch.full((1, 1), clamp_min, device=self.device)
            else:
                logger.warning("No scores available in generation output")
                gen_logps = torch.full((1, max(1, generated_tokens.shape[1])), clamp_min, device=self.device)
            
            # 最终安全检查
            gen_logps = self._clean_extreme_values(gen_logps, "token log probabilities")
            
            return gen_logps
            
        except Exception as e:
            logger.error(f"Error computing generation log probabilities: {e}")
            # 返回安全的默认值
            seq_len = max(1, generated_tokens.shape[1] if generated_tokens is not None else 1)
            return torch.full((1, seq_len), clamp_min, device=self.device)
    
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

