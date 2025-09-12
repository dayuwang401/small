import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from torch.nn.utils.rnn import pad_sequence
import logging
import re
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SYSTEM_PROMPT_R = """Please integrate natural language reasoning with programs to solve the problem, and put your final answer within \boxed{}
""" 


USER_PROMPT_TEMPLATE = """Problem:
{question}
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

class ReasoningAgent_v2:
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
        """重新加载最新权重（RL 更新后可调用）"""
        self.tokenizer, self.model = self._load_model(self.model_path, self.device, self.dtype)

    def build_prompt(self, question):
        return f"{SYSTEM_PROMPT_R}\n\n" + USER_PROMPT_TEMPLATE.format(question=question)

    def generate_until_code_or_answer(self, prompt, n_samples, temperature=0.7, top_p=0.95, max_new_tokens=512):
        """Generate text with batch parallel generation for multiple samples,
           and return generated_texts, gen_logps, outputs.sequences (padded to max length)."""
        self.model.eval()
        logger.debug("Starting batch text generation")
        
        try:
            # 创建batch输入：将prompt重复n_samples次
            prompts = [prompt] * n_samples
            logger.debug(f"Creating batch input with {n_samples} samples")
            
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            logger.debug(f"Batch input tokenized, shape: {inputs['input_ids'].shape}")
            
            input_len = inputs["input_ids"].shape[1]
            logger.debug(f"Input length: {input_len} tokens")
            
            logger.debug("Starting batch generation")
            
            with torch.no_grad():
                try:
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
                    logger.debug("Batch generation completed successfully")
                    
                    # 生成部分 tokens (batch_size, generated_seq_len)
                    generated_tokens = outputs.sequences[:, input_len:]
                    logger.debug(f"Generated tokens shape: {generated_tokens.shape}")
                    
                    # 解码生成的文本
                    generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
                    logger.debug(f"Decoded {len(generated_texts)} generated texts")
                    
                    # 计算 log probs - batch版本
                    if hasattr(outputs, 'scores') and outputs.scores:
                        logger.debug(f"Computing log probabilities for batch with {len(outputs.scores)} score tensors")
                        
                        batch_gen_logps = []
                        
                        for i, scores in enumerate(outputs.scores):
                            if i < generated_tokens.shape[1]:
                                # 确保数值稳定性
                                scores = scores.float()
                                
                                # 预处理scores，避免极端值
                                scores = torch.clamp(scores, min=-50.0, max=50.0)
                                
                                # 计算log_softmax
                                log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
                                
                                # 温和的极值处理
                                log_probs = torch.clamp(log_probs, min=-10.0, max=0.0)
                                
                                # 处理可能的NaN或Inf
                                if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                                    logger.warning(f"NaN/Inf detected in log_probs at step {i}, replacing with safe values")
                                    log_probs = torch.nan_to_num(log_probs, nan=-10.0, neginf=-10.0, posinf=0.0)
    
                                # 获取每个样本对应token的log probability
                                selected_logps = torch.gather(log_probs, dim=-1, 
                                                              index=generated_tokens[:, i].unsqueeze(-1)).squeeze(-1)  # (batch_size,)
                                batch_gen_logps.append(selected_logps)
                        
                        logger.debug(f"Log probabilities computed for batch")
                        
                        if batch_gen_logps:
                            # 转换为 (batch_size, seq_len)
                            all_gen_logps = torch.stack(batch_gen_logps, dim=1)
                        else:
                            logger.warning("No log probabilities computed, using zeros")
                            all_gen_logps = torch.zeros((n_samples, 1), device=self.device)
                    else:
                        logger.warning("No scores available, using zero log probabilities")
                        all_gen_logps = torch.zeros((n_samples, 1), device=self.device)
                    
                    # outputs.sequences 已经是 (batch_size, total_seq_len) 格式
                    all_sequences = outputs.sequences
                    logger.debug(f"Final sequences shape: {all_sequences.shape}")
                    
                except Exception as e:
                    logger.error(f"Error in batch generation: {e}")
                    # 创建安全的默认值
                    generated_texts = ["Error in generation"] * n_samples
                    all_gen_logps = torch.zeros((n_samples, 1), device=self.device)
                    all_sequences = inputs["input_ids"]  # 返回原始输入
            
            # 最终验证输出
            if len(generated_texts) != n_samples:
                logger.warning(f"Generated {len(generated_texts)} texts but expected {n_samples}")
            
            if all_gen_logps.shape[0] != n_samples:
                logger.warning(f"Generated log probs shape {all_gen_logps.shape} but expected {n_samples} samples")
            
            if all_sequences.shape[0] != n_samples:
                logger.warning(f"Generated sequences shape {all_sequences.shape} but expected {n_samples} samples")
            
            # 最终检查log_probs的有效性
            if torch.isnan(all_gen_logps).any() or torch.isinf(all_gen_logps).any():
                logger.warning("Invalid log probabilities detected in final output, replacing with safe values")
                all_gen_logps = torch.clamp(torch.nan_to_num(all_gen_logps, nan=-10.0, neginf=-10.0, posinf=0.0), 
                                          min=-10.0, max=0.0)
            
            logger.info(f"Batch generation completed: {len(generated_texts)} texts, "
                       f"log_probs shape: {all_gen_logps.shape}, sequences shape: {all_sequences.shape}")
            
            return generated_texts, all_gen_logps, all_sequences
            
        except Exception as e:
            logger.error(f"Critical error in batch text generation: {e}")
            # 返回安全的默认值
            dummy_texts = [""] * n_samples
            dummy_logps = torch.zeros((n_samples, 1), device=self.device)
            dummy_sequences = torch.zeros((n_samples, 10), dtype=torch.long, device=self.device)
            
            return dummy_texts, dummy_logps, dummy_sequences
