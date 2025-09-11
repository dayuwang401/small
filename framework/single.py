

import torch
import torch.multiprocessing as mp
from torch import optim
from agents.reasoning_agent import ReasoningAgent
from agents.code_agent import CodeAgent
from reward import reward_code_execution
from trainer import Trainer
import os
import re
import copy
import argparse
import json
import gc
import logging
import re
import random
import requests

SANDBOX_URL = "http://localhost:8080/run_code"
def extract_boxed_content(text):
    results = []
    i = 0
    while i < len(text):
        # 查找 \boxed{
        if text[i:i+6] == r'boxed{':
            i += 6
            brace_count = 1
            start = i
            while i < len(text) and brace_count > 0:
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                i += 1
            # 去掉最外层的 {} 对
            content = text[start:i-1]
            results.append(content)
        else:
            i += 1
    return results
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def run_code_in_sandbox(code_str):
    """执行 Python 代码并返回 stdout 或错误"""
    try:
        resp = requests.post(SANDBOX_URL, json={"code": code_str, "language": "python"}, timeout=20)
    except Exception as e:
        return False
    try:
        res = resp.json()
        # print(res)  # 可调试返回结果
    except Exception as e:
        return False
    
    if res.get("status") == "Success":
        return res["run_result"]["stdout"].strip()
    else:
        return False
def reward_code_execution(code_str, ground_truth_list,return_result=False):
    """
    执行代码并根据输出和ground truth匹配程度给奖励。
    
    参数:
        code_str (str): 待执行的Python代码
        ground_truth_list (list[str]): 正确输出的可能值列表（按字符串精确匹配）
    
    返回:
        float: 奖励分数 (1.0 = 正确, 0.2 = 成功但不正确, 0.0 = 失败)
    """
    output = run_code_in_sand(code_str)
    if return_result:
        if output:
            return output
        else:
            return "```"
    print("code",code_str)
    
    # 如果运行失败
    if output is False:
        print("reward",0)
        return 0.0

    # 正确匹配 ground truth
    if output in ground_truth_list:
        print("reward",1.0)
        return 1.0

    # 成功执行但结果不在正确集合中
    print("reward",0.2)
    return 0.2

def extract_last_code_block(text):
    """提取最后一个的内容（保留换行）"""
    matches = re.findall(r"```python\s*([\s\S]*?)```", text, re.DOTALL)
    if matches:
        return matches[-1]  # 取最后一个
    return "print("

def get_per_token_logps(logits, input_ids, clamp_min=-5.0, clamp_max=0.0):
    """
    计算每个token的log概率，并进行极值裁剪
    
    Args:
        logits: 模型输出的logits [batch_size, seq_len, vocab_size]
        input_ids: 输入的token ids [batch_size, seq_len]
        clamp_min: log概率的最小值，防止数值下溢
        clamp_max: log概率的最大值，理论上log概率应该 <= 0
    
    Returns:
        per_token_logps: 每个token的log概率 [batch_size, seq_len]
    """
    # 确保使用float32进行计算，提高数值稳定性
    logits = logits.float()
    
    # 计算log_softmax，内部已经有数值稳定性处理
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # 提取对应token的log概率
    per_token_logps = torch.gather(log_probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    
    # 极值裁剪，防止数值不稳定
    per_token_logps = torch.clamp(per_token_logps, min=clamp_min, max=clamp_max)
    
    # 可选：检查是否有异常值
    if torch.isnan(per_token_logps).any():
        print("Warning: NaN detected in per_token_logps after clamping")
        per_token_logps = torch.nan_to_num(per_token_logps, nan=clamp_min)
    
    if torch.isinf(per_token_logps).any():
        print("Warning: Inf detected in per_token_logps after clamping")
        per_token_logps = torch.clamp(per_token_logps, min=clamp_min, max=clamp_max)
    
    return per_token_logps

def GRPO_step(model, tokenizer, batch, beta=0.001, clip_param=0.2):
    """GRPO训练步骤"""
    print("开始grpo loss计算")
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(next(model.parameters()).device)
    advantages = batch['rewards'].unsqueeze(1) if len(batch['rewards'].shape) == 1 else batch['rewards']
    
    # 前向传播
    logits = model(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V)
    input_ids = inputs[:, 1:]   # (B, L-1)
    
    # 计算当前模型的log概率
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:, prompt_length-1:]
    
    # 参考模型的log概率
    
    
    # KL散度
    
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    
    # GRPO损失
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'])
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
    
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    print("结束grpo loss计算")
    return loss

class GRPOTrainer:
    def __init__(self, reasoning_model_path, code_model_path):
        # 检查GPU可用性
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        gpu_count = torch.cuda.device_count()
        if gpu_count < 2:
            raise RuntimeError(f"Need at least 2 GPUs but only {gpu_count} available")

        # 固定设备分配：reasoning agent在GPU 0，code agent在GPU 1
        self.reasoning_device = torch.device("cuda:0")
        self.code_device = torch.device("cuda:1")

        # 初始化模型
        try:
            # 推理模型及其副本
            self.base_reasoning_model = ReasoningAgent(reasoning_model_path, device=self.reasoning_device, dtype=torch.float32).model
            self.reasoning_agent = ReasoningAgent(reasoning_model_path, device=self.reasoning_device, dtype=torch.float32)
            self.reasoning_agent.model = copy.deepcopy(self.base_reasoning_model)

            # 代码模型及其副本
            self.base_code_model = CodeAgent(code_model_path, device=self.code_device, dtype=torch.float32).model
            self.code_agent = CodeAgent(code_model_path, device=self.code_device, dtype=torch.float32)
            self.code_agent.model = copy.deepcopy(self.base_code_model)
            
            logger.info("Models initialized successfully")
            logger.info(f"Reasoning model on: {self.reasoning_device}")
            logger.info(f"Code model on: {self.code_device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

        # 参考模型（用于GRPO，不需要梯度）
        

        # 冻结参考模型参数
       

        # 优化器
        self.reasoning_optimizer = optim.AdamW(self.reasoning_agent.model.parameters(), lr=1e-6, eps=1e-8)
        self.code_optimizer = optim.AdamW(self.code_agent.model.parameters(), lr=1e-6, eps=1e-8)

    


    def sample_reasonings(self, prompt, n_samples, max_tokens=2048):
        """采样推理文本"""
        print("采样推理文本")
        try:
            texts, gen_logps, sequences = self.reasoning_agent.generate_until_code_or_answer(
                prompt , n_samples
                
            )
            
            torch.cuda.empty_cache()
            print("推理文本采样完成")
            return texts, gen_logps, sequences
            
        except Exception as e:
            logger.error(f"Error in sample_reasonings: {e}")
            dummy_texts = ["print(0"] * n_samples
            dummy_logps = torch.zeros((n_samples, 1), device=self.reasoning_device)
            dummy_seqs = torch.zeros((n_samples, 10), dtype=torch.long, device=self.reasoning_device)
            return dummy_texts, dummy_logps, dummy_seqs

    def sample_codes(self, prompt, n_samples, max_tokens=1024):
        """采样代码文本"""
        print("采样代码文本")
        try:
            texts, gen_logps, sequences = self.code_agent.generate_until_code_end(
                prompt , n_samples
            )
            
            torch.cuda.empty_cache()
            return texts, gen_logps, sequences
            
        except Exception as e:
            logger.error(f"Error in sample_codes: {e}")
            dummy_texts = ["print(0"] * n_samples
            dummy_logps = torch.zeros((n_samples, 1), device=self.code_device)
            dummy_seqs = torch.zeros((n_samples, 10), dtype=torch.long, device=self.code_device)
            return dummy_texts, dummy_logps, dummy_seqs

    
        
    def run_step(self, history, truth, n_reasoning=6, n_code=6, beta=0):
        """运行一步训练"""
        logger.info("Starting training step")
    
        try:
            # 保存模型状态用于可能的回滚
            reasoning_state_backup = copy.deepcopy(self.reasoning_agent.model.state_dict())
            code_state_backup = copy.deepcopy(self.code_agent.model.state_dict())
    
            # 1. 生成推理样本
            logger.debug("Generating reasoning samples")
            reasoning_prompt = self.reasoning_agent.build_prompt(history)
            reasoning_texts, reasoning_gen_logps, reasoning_sequences = self.sample_reasonings(
                reasoning_prompt, n_reasoning
            )
    
            # 计算推理prompt长度
            reasoning_prompt_tokens = self.reasoning_agent.tokenizer(
                reasoning_prompt, return_tensors="pt"
            ).input_ids.shape[1]
    
            # 存储所有批次数据和奖励
            code_batches = []
            reasoning_rewards = []
            reasoning_code_map = []
            best_reasoning_result = None
    
            # if any("boxed" in text for text in reasoning_texts):
            #     logger.info("Answer found in reasoning texts, stopping")
            #     return torch.tensor(0.0, device=self.reasoning_device, requires_grad=True), torch.tensor(0.0, device=self.code_device, requires_grad=True), "boxed"
    
            # 2. 对每个推理生成代码样本
            logger.debug(f"Processing {len(reasoning_texts)} reasoning texts")
            for i, r_text in enumerate(reasoning_texts):
                try:
                    logger.debug(f"Processing reasoning sample {i+1}/{len(reasoning_texts)}")
                    if "boxed" in r_text:
                        if extract_boxed_content(r_text) in turth:
                            reasoning_rewards.append(1)
                        else:
                            reasoning_rewards.append(0)
                        continue
                    r_text_clean = re.sub(r"```python", "", r_text, flags=re.DOTALL)
                    code_prompt = self.code_agent.build_prompt(r_text_clean)
                    code_texts, code_gen_logps, code_sequences = self.sample_codes(code_prompt, n_code)
    
                    # 计算代码prompt长度
                    code_prompt_tokens = self.code_agent.tokenizer(
                        code_prompt, return_tensors="pt"
                    ).input_ids.shape[1]
    
                    # 计算代码奖励
                    rewards_code = []
                    for j, c in enumerate(code_texts):
                        try:
                            reward = reward_code_execution(extract_last_code_block(c), truth)
                            rewards_code.append(reward)
                        except Exception as e:
                            logger.warning(f"Code execution failed for sample {j}: {e}")
                            rewards_code.append(0.0)
    
                    rewards_code = torch.tensor(rewards_code, dtype=torch.float32, device=self.code_device)
                    logger.debug(f"Code rewards computed: mean={rewards_code.mean():.4f}, std={rewards_code.std():.4f}")
    
                    # 记录推理奖励（代码平均奖励）
                    reasoning_rewards.append(rewards_code.mean().item())
    
                    # 检查代码reward标准差是否大于0.1
                    if rewards_code.std() > 0.1:
                        logger.debug("Code reward std > 0.1, entering GRPO process")
    
                        # 归一化奖励
                        rewards_code = (rewards_code - rewards_code.mean()) / (rewards_code.std() + 1e-8)
    
                        # 获取参考模型的log概率
                        
    
                        # 构建批次数据
                        if rewards_code.numel() > 0 and not torch.allclose(rewards_code, rewards_code[0], atol=1e-6):
                            logger.debug(f"Code rewards: {rewards_code}")
                            batch = {
                                'plen': code_prompt_tokens,
                                'inputs': code_sequences,
                                'rewards': rewards_code,
                                'gen_logps': code_gen_logps
                            }
                            code_batches.append(batch)
    
                        
                    else:
                        logger.debug("Code reward std <= 0.1, skipping GRPO process")
    
                    avg_reward = rewards_code.mean().item()
                    logger.debug(f"Average reward for reasoning {i}: {avg_reward:.4f}")
                    reasoning_code_map.append((r_text, list(zip(code_texts, rewards_code.tolist()))))
    
                    if best_reasoning_result is None or avg_reward > best_reasoning_result['reward']:
                        best_code_idx = rewards_code.argmax().item()
                        best_code = code_texts[best_code_idx]
    
                        try:
                            best_code_result = reward_code_execution(
                                extract_last_code_block(best_code), truth, True
                            )
                        except:
                            best_code_result = "Error"
    
                        best_reasoning_result = {
                            'reward': avg_reward,
                            'reasoning_text': r_text,
                            'best_code': best_code,
                            'best_code_reward': rewards_code[best_code_idx].item(),
                            'best_code_result': best_code_result
                        }
    
                    # 清理临时张量
                    del rewards_code
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
    
                except Exception as e:
                    logger.error(f"Error processing reasoning {i}: {e}")
                    reasoning_rewards.append(0.0)
                    reasoning_code_map.append((r_text, [("print(", 0.0)]))
            if any("boxed" in text for text in reasoning_texts):
                history="boxed"
            else:
                history = history + best_reasoning_result['reasoning_text'] + "```\n```output```\n" + best_reasoning_result['best_code_result'] + "\n```"
            
            logger.debug(f"Updated history length: {len(history)}")
            logger.debug(f"Reasoning rewards: {reasoning_rewards}")
    
            # 3. 更新代码模型
            
            
            self.code_optimizer.zero_grad()
            if code_batches:
                logger.info("Starting code model update")
                self.code_agent.model.train()
    
                for batch in code_batches:
                    try:
                        loss = GRPO_step(
                            self.code_agent.model, 
                            self.code_agent.tokenizer, batch, beta=beta
                        )
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            (loss/n_reasoning).backward()
                    except Exception as e:
                        code_loss = torch.tensor(0.0, device=self.code_device, requires_grad=True)
            else:
                logger.debug("No code batches with std > 0.1, skipping code model update")
            if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            code_loss = torch.tensor(0.0, device=self.code_device, requires_grad=True)
            # 4. 更新推理模型
            reasoning_loss = None
            logger.debug(f"Reasoning rewards: {reasoning_rewards}")
    
            if reasoning_rewards:
                reasoning_rewards_tensor = torch.tensor(reasoning_rewards, device=self.reasoning_device)
    
                if reasoning_rewards_tensor.std() > 0.1:
                    logger.info("Reasoning reward std > 0.1, entering GRPO process")
                    logger.debug("Starting reasoning model update")
    
                    if reasoning_rewards_tensor.std() > 1e-8:
                        reasoning_rewards_tensor = (reasoning_rewards_tensor - reasoning_rewards_tensor.mean()) / (reasoning_rewards_tensor.std() + 1e-8)
    
                    if reasoning_rewards_tensor.numel() > 1:
                        try:
                            reasoning_batch = {
                                'plen': reasoning_prompt_tokens,
                                'inputs': reasoning_sequences,
                                'rewards': reasoning_rewards_tensor,
                                
                                'gen_logps': reasoning_gen_logps
                            }
                            reasoning_loss = GRPO_step(
                                self.reasoning_agent.model, 
                                self.reasoning_agent.tokenizer, reasoning_batch, beta=beta
                            )
                        except Exception as e:
                            logger.error(f"Error computing reasoning loss: {e}")
                            del reasoning_rewards_tensor
                            reasoning_update_success = False
                else:
                    logger.debug("Reasoning reward std <= 0.1, skipping reasoning model update")
                    del reasoning_rewards_tensor
    
            del reasoning_rewards, reasoning_code_map, reasoning_texts, reasoning_gen_logps, reasoning_sequences
            del reasoning_state_backup, code_state_backup
    
            torch.cuda.empty_cache()
    
            logger.info(f"Training step completed - Code update: {'Success' if code_update_success else 'Failed'}, "
                        f"Reasoning update: {'Success' if reasoning_update_success else 'Failed'}")
    
            return reasoning_loss, code_loss, history
    
        except Exception as e:
            logger.error(f"Error in run_step: {e}")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            raise e
    def _check_gradients(self, model, model_name):
        """
        检查模型梯度是否包含极端值
        
        Args:
            model: 要检查的模型
            model_name: 模型名称，用于日志
            
        Returns:
            bool: True表示梯度正常，False表示包含极端值
        """
        has_nan = False
        has_inf = False
        max_grad = 0.0
        grad_count = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_data = param.grad.data
                grad_count += 1
                
                if torch.isnan(grad_data).any():
                    has_nan = True
                    logger.error(f"NaN gradient detected in {model_name} parameter: {name}")
                
                if torch.isinf(grad_data).any():
                    has_inf = True
                    logger.error(f"Inf gradient detected in {model_name} parameter: {name}")
                
                grad_norm = grad_data.norm().item()
                max_grad = max(max_grad, grad_norm)
                
                # 检查梯度是否过大
                if grad_norm > 100.0:
                    logger.warning(f"Large gradient norm ({grad_norm:.2f}) in {model_name} parameter: {name}")
        
        if grad_count == 0:
            logger.warning(f"No gradients found in {model_name}")
            return False
        
        logger.debug(f"{model_name} gradient check - Count: {grad_count}, Max norm: {max_grad:.6f}, "
                    f"Has NaN: {has_nan}, Has Inf: {has_inf}")
        
        if has_nan or has_inf:
            logger.error(f"{model_name} gradients contain extreme values")
            return False
        
        return True
        
    def _check_model_parameters(self, model, model_name):
        """
        检查模型参数是否包含极端值
        
        Args:
            model: 要检查的模型
            model_name: 模型名称，用于日志
            
        Returns:
            bool: True表示参数正常，False表示包含极端值
        """
        has_nan = False
        has_inf = False
        max_param = 0.0
        param_count = 0
        
        for name, param in model.named_parameters():
            if param.data is not None:
                param_data = param.data
                param_count += 1
                
                if torch.isnan(param_data).any():
                    has_nan = True
                    # logger.error(f"NaN parameter detected in {model_name} parameter: {name}")
                
                if torch.isinf(param_data).any():
                    has_inf = True
                    # logger.error(f"Inf parameter detected in {model_name} parameter: {name}")
                
                param_norm = param_data.norm().item()
                max_param = max(max_param, param_norm)
                
                # 检查参数是否过大
                if param_norm > 1000.0:
                    logger.warning(f"Large parameter norm ({param_norm:.2f}) in {model_name} parameter: {name}")
        
        logger.debug(f"{model_name} parameter check - Count: {param_count}, Max norm: {max_param:.6f}, "
                    f"Has NaN: {has_nan}, Has Inf: {has_inf}")
        
        if has_nan or has_inf:
            logger.error(f"{model_name} parameters contain extreme values")
            return False
        
        return True
    def save_checkpoint(self, step, reasoning_model_path, code_model_path):
        """保存检查点"""
        try:
            save_dir_r = os.path.join(reasoning_model_path, f"checkpoint_step_{step}")
            save_dir_c = os.path.join(code_model_path, f"checkpoint_step_{step}")
            
            # 创建保存目录
            os.makedirs(save_dir_r, exist_ok=True)
            os.makedirs(save_dir_c, exist_ok=True)
            
            # 保存推理模型和tokenizer
            self.reasoning_agent.model.save_pretrained(save_dir_r)
            self.reasoning_agent.tokenizer.save_pretrained(save_dir_r)
            
            # 保存代码模型和tokenizer
            self.code_agent.model.save_pretrained(save_dir_c)
            self.code_agent.tokenizer.save_pretrained(save_dir_c)
            
            # 保存训练状态
            checkpoint_info = {
                'step': step,
                'reasoning_optimizer_state': self.reasoning_optimizer.state_dict(),
                'code_optimizer_state': self.code_optimizer.state_dict(),
            }
            
            torch.save(checkpoint_info, os.path.join(save_dir_r, 'training_state.pt'))
            
            logger.info(f"Saved checkpoint at Step {step}")
            logger.info(f"Reasoning model saved to: {save_dir_r}")
            logger.info(f"Code model saved to: {save_dir_c}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint at step {step}: {e}")

    def load_checkpoint(self, checkpoint_step, reasoning_model_path, code_model_path):
        """加载检查点"""
        try:
            save_dir_r = os.path.join(reasoning_model_path, f"checkpoint_step_{checkpoint_step}")
            save_dir_c = os.path.join(code_model_path, f"checkpoint_step_{checkpoint_step}")
            
            if not (os.path.exists(save_dir_r) and os.path.exists(save_dir_c)):
                raise FileNotFoundError(f"Checkpoint directories not found: {save_dir_r}, {save_dir_c}")
            
            # 加载训练状态
            training_state_path = os.path.join(save_dir_r, 'training_state.pt')
            if os.path.exists(training_state_path):
                checkpoint_info = torch.load(training_state_path, map_location='cpu')
                self.reasoning_optimizer.load_state_dict(checkpoint_info['reasoning_optimizer_state'])
                self.code_optimizer.load_state_dict(checkpoint_info['code_optimizer_state'])
                logger.info(f"Loaded training state from step {checkpoint_info['step']}")
            
            logger.info(f"Checkpoint loaded successfully from step {checkpoint_step}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from step {checkpoint_step}: {e}")
            raise

def load_jsonl(path):
    """加载 JSONL 文件"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(l.strip()) for l in f if l.strip()]
        
        if not lines:
            raise ValueError(f"No valid data found in {path}")
            
        logger.info(f"Loaded {len(lines)} questions from {path}")
        return lines
        
    except Exception as e:
        logger.error(f"Failed to load JSONL file {path}: {e}")
        raise

def run_training(reasoning_model_path, code_model_path, jsonl_path, save_every_update, resume_step=None):
    """单进程训练主函数，code每步更新一次，reasoning每10步更新一次，按更新次数存ckpt"""
    try:
        trainer = GRPOTrainer(reasoning_model_path, code_model_path)
        
        # 恢复进度
        reasoning_update_count = 0
        if resume_step is not None and resume_step > 0:
            trainer.load_checkpoint(resume_step, reasoning_model_path, code_model_path)
            reasoning_update_count = resume_step
            logger.info(f"Resuming from reasoning update {reasoning_update_count}")

        # Reasoning梯度累积的步数
        reasoning_accum_steps = 10
        reasoning_step_counter = 0

        # 加载数据
        questions = load_jsonl(jsonl_path)

        for q_idx, item in enumerate(questions):
            question = item.get("question", "")
            global_history = question
            logger.info(f"Starting Q{q_idx} prompt: {global_history[:50]}...")
            truth = item.get("mid_ground_truth", "")

            for step in range(3):  # 最多 3 步
                try:
                    # run_step 内部已经采样了 8 次 reasoning，并生成了 code_loss_list
                    # code_loss_list: list of losses (1 per reasoning sample)
                    code_loss, reasoning_loss, global_history = trainer.run_step(global_history, truth)
                    
                    # 1️⃣ Code 部分 —— 累积梯度到一次 step
                    trainer.code_optimizer.step()       # 一次性更新 code 模型参数
                    trainer.code_optimizer.zero_grad()
                    # 2️⃣ Reasoning 部分 —— 梯度累积，每10步更新一次
                    (reasoning_loss / reasoning_accum_steps).backward()
                    reasoning_step_counter += 1

                    if reasoning_step_counter % reasoning_accum_steps == 0:
                        trainer.reasoning_optimizer.step()
                        trainer.reasoning_optimizer.zero_grad()
                        reasoning_update_count += 1
                        logger.info(f"Reasoning updated {reasoning_update_count} times.")

                        # 保存 checkpoint
                        if reasoning_update_count % save_every_update == 0:
                            trainer.save_checkpoint(reasoning_update_count, reasoning_model_path, code_model_path)

                    logger.info(
                        f"Q{q_idx} Step {step} "
                        f"Loss_R={reasoning_loss.item():.4f} "
                       
                    )

                    if history==

                except Exception as e:
                    logger.error(f"Error in Q{q_idx} Step {step}: {e}")
                    break

        logger.info(f"Training completed. Reasoning updates: {reasoning_update_count}")

        # 最终保存
        trainer.save_checkpoint(reasoning_update_count, reasoning_model_path, code_model_path)

    except Exception as e:
        logger.error(f"Critical error in training: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Single Process Training")
    parser.add_argument("--reasoning_model_path", type=str, default="gpt2", 
                       help="Path to reasoning model")
    parser.add_argument("--code_model_path", type=str, default="gpt2",
                       help="Path to code model")
    parser.add_argument("--jsonl_path", type=str, required=True,
                       help="Path to JSONL file with training questions")
    parser.add_argument("--save_every", type=int, default=10,
                       help="Save checkpoint every N steps")
    parser.add_argument("--resume_step", type=int, default=None,
                       help="Resume from specific checkpoint step")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 验证参数
    if not os.path.exists(args.jsonl_path):
        raise FileNotFoundError(f"JSONL file not found: {args.jsonl_path}")
    
    # 检查GPU数量
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        logger.warning(f"Only {gpu_count} GPUs available, but 2 required for optimal performance")
    
    logger.info("Starting single process training")
    logger.info(f"Reasoning model will use GPU 0")
    logger.info(f"Code model will use GPU 1")
    
    # 启动训练
    run_training(
        args.reasoning_model_path, 
        args.code_model_path, 
        args.jsonl_path, 
        args.save_every, 
        args.resume_step
    )
    
    logger.info("Training completed successfully!")
