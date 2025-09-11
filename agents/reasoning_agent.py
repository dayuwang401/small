import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from torch.nn.utils.rnn import pad_sequence
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
SYSTEM_PROMPT_R = """Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \boxed{}
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

class ReasoningAgent:
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
        """Generate text with keyword stopping for a single prompt, multiple samples (variable length OK),
           and return generated_texts, gen_logps, outputs.sequences (padded to max length)."""
        self.model.eval()
        logger.debug("Starting text generation")
        
        try:
            # 编码单条 prompt
            logger.debug("Tokenizing input prompt")
            inputs = self.tokenizer(
                [prompt], 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            logger.debug(f"Input tokenized, shape: {inputs['input_ids'].shape}")
            
            input_len = inputs["input_ids"].shape[1]
            logger.debug(f"Input length: {input_len} tokens")
            
            stopping_criteria = StoppingCriteriaList([
                    KeywordStoppingCriteria(self.tokenizer, keywords=[ r"```python"], start_length=input_len)
                ])
            logger.debug("Stopping criteria initialized")
            
            all_generated_texts = []
            all_gen_logps_list = []
            all_sequences_list = []
            logger.debug(f"Starting generation for {n_samples} samples")
            
            with torch.no_grad():
                for sample_idx in range(n_samples):
                    logger.debug(f"Generating sample {sample_idx + 1}/{n_samples}")
                    
                    try:
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
                        
                        # 生成部分 tokens
                        generated_tokens = outputs.sequences[:, input_len:]
                        generated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
                        all_generated_texts.extend(generated_text)
                        logger.debug(f"Sample {sample_idx + 1} decoded, generated {generated_tokens.shape[1]} tokens")
                        
                        # 计算 log probs - 使用更稳定的方法
                        gen_logps = []
                        if hasattr(outputs, 'scores') and outputs.scores:
                            logger.debug(f"Computing log probabilities for {len(outputs.scores)} score tensors")
                            
                            for i, scores in enumerate(outputs.scores):
                                if i < generated_tokens.shape[1]:
                                    # 确保数值稳定性
                                    scores = scores.float()
                                    
                                    # 预处理scores，避免极端值
                                    scores = torch.clamp(scores, min=-50.0, max=50.0)
                                    
                                    # 计算log_softmax，内部已有数值稳定性处理
                                    log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
                                    
                                    # 温和的极值处理，使用更合理的范围
                                    log_probs = torch.clamp(log_probs, min=-10.0, max=0.0)
                                    
                                    # 处理可能的NaN或Inf
                                    if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
                                        logger.warning(f"NaN/Inf detected in log_probs at step {i}, replacing with safe values")
                                        log_probs = torch.nan_to_num(log_probs, nan=-10.0, neginf=-10.0, posinf=0.0)
            
                                    selected_logps = torch.gather(log_probs, dim=-1, 
                                                                  index=generated_tokens[:, i].unsqueeze(-1)).squeeze(-1)
                                    gen_logps.append(selected_logps)
                            
                            logger.debug(f"Log probabilities computed for sample {sample_idx + 1}")
                            
                            if gen_logps:
                                gen_logps = torch.stack(gen_logps, dim=1)  # (1, seq_len)
                            else:
                                logger.warning(f"No log probabilities computed for sample {sample_idx + 1}, using zeros")
                                gen_logps = torch.zeros((1, 1), device=self.device)
                        else:
                            logger.warning(f"No scores available for sample {sample_idx + 1}, using zero log probabilities")
                            gen_logps = torch.zeros((1, 1), device=self.device)
            
                        all_gen_logps_list.append(gen_logps.squeeze(0))  # 去掉 batch 维度
                        all_sequences_list.append(outputs.sequences.squeeze(0))  # (seq_total_len,)
                        
                        logger.debug(f"Sample {sample_idx + 1} processing completed")
                        
                    except Exception as e:
                        logger.error(f"Error generating sample {sample_idx + 1}: {e}")
                        # 创建一个最小的有效输出来保持批次一致性
                        dummy_text = "Error in generation"
                        all_generated_texts.append(dummy_text)
                        
                        dummy_logps = torch.zeros((1,), device=self.device)
                        all_gen_logps_list.append(dummy_logps)
                        
                        dummy_sequence = torch.cat([inputs["input_ids"].squeeze(0), 
                                                  torch.tensor([self.tokenizer.eos_token_id], device=self.device)])
                        all_sequences_list.append(dummy_sequence)
            
            logger.debug("All samples generated, starting padding")
            
            # 对不同长度进行 padding，并检查有效性
            if all_gen_logps_list:
                # 检查log_probs的有效性
                valid_logps = []
                for i, logps in enumerate(all_gen_logps_list):
                    if torch.isnan(logps).any() or torch.isinf(logps).any():
                        logger.warning(f"Invalid log probabilities detected in sample {i}, replacing with safe values")
                        logps = torch.clamp(torch.nan_to_num(logps, nan=-10.0, neginf=-10.0, posinf=0.0), 
                                          min=-10.0, max=0.0)
                    valid_logps.append(logps)
                
                all_gen_logps = pad_sequence(valid_logps, batch_first=True, padding_value=-10.0)  # 使用更合理的padding值
                logger.debug(f"Log probabilities padded to shape: {all_gen_logps.shape}")
            else:
                logger.warning("No valid log probabilities generated, using zeros")
                all_gen_logps = torch.zeros((n_samples, 1), device=self.device)
            
            if all_sequences_list:
                all_sequences = pad_sequence(all_sequences_list, batch_first=True, 
                                           padding_value=self.tokenizer.pad_token_id)
                logger.debug(f"Sequences padded to shape: {all_sequences.shape}")
            else:
                logger.warning("No valid sequences generated, using dummy sequences")
                all_sequences = inputs["input_ids"].repeat(n_samples, 1)
            
            # 最终验证输出
            if len(all_generated_texts) != n_samples:
                logger.warning(f"Generated {len(all_generated_texts)} texts but expected {n_samples}")
            
            if all_gen_logps.shape[0] != n_samples:
                logger.warning(f"Generated log probs shape {all_gen_logps.shape} but expected {n_samples} samples")
            
            if all_sequences.shape[0] != n_samples:
                logger.warning(f"Generated sequences shape {all_sequences.shape} but expected {n_samples} samples")
            
            logger.info(f"Generation completed: {len(all_generated_texts)} texts, "
                       f"log_probs shape: {all_gen_logps.shape}, sequences shape: {all_sequences.shape}")
            
            return all_generated_texts, all_gen_logps, all_sequences
            
        except Exception as e:
            logger.error(f"Critical error in text generation: {e}")
            # 返回安全的默认值
            dummy_texts = [""] * n_samples
            dummy_logps = torch.zeros((n_samples, 1), device=self.device)
            dummy_sequences = inputs["input_ids"].repeat(n_samples, 1) if 'inputs' in locals() else torch.zeros((n_samples, 10), dtype=torch.long, device=self.device)
            
            return dummy_texts, dummy_logps, dummy_sequences
