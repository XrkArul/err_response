# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import math
import torch
import torch.nn.functional as F
import numpy as np

def extract_solution(solution_str, method='strict'):
    """
    Extract the final numerical answer from a solution string.
    
    Args:
        solution_str (str): The complete solution text
        method (str): Extraction method - 'strict' for formatted answers (####), 
                     'flexible' for any numerical value
    
    Returns:
        str or None: The extracted answer, or None if no valid answer found
    """
    assert method in ['strict', 'flexible']

    if method == 'strict':
        # Extract answer in format "#### NUMBER" (GSM8k standard format)
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split('#### ')[1].replace(',', '').replace('$', '')
    
    elif method == 'flexible':
        # Find all numbers in the solution string
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            pass
        else:
            # Extract the last valid number (excluding empty strings and lone periods)
            invalid_str = ['', '.']
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    
    # Fallback: try to extract answer from LaTeX \boxed{} format
    if final_answer is None:
        match = re.search(r'\\boxed\{(.*?)\}', solution_str)
        if match:
            final_answer = match.group(1).strip()
    
    return final_answer


def compute_score(solution_str, ground_truth, old_log_prob, valid_response_ids,
                       method='strict', format_score=0., score=1., tokenizer=None,
                       group_lengths=None, extra_info=None):
    """
    Based on Robust Relative Efficiency Reward (RRER).
    Completely decouples task score from length penalty to avoid gradient clipping traps.
    """
    # === 0. Determine if in test/validation mode ===
    # Test/validation mode only returns base score (1.0 or 0.0), without any additional rewards
    is_test_mode = False
    if extra_info is not None:
        split = extra_info.get("split", "train")
        if split == "test":
            is_test_mode = True

    # === 1. Base Task Score & Entropy Relief ===
    answer = extract_solution(solution_str=solution_str, method=method)
    is_correct = (answer is not None and answer == ground_truth)
    base_score = score if is_correct else format_score

    # Test/validation mode directly returns base score
    if is_test_mode:
        return base_score

    relief_reward = 0.0
    epsilon_err = 0.01   
    lambda_err = 0.3
    
    if old_log_prob.get("entropys") is not None and tokenizer is not None:
        entropys = old_log_prob["entropys"]
        token_strs = tokenizer.convert_ids_to_tokens(valid_response_ids)
        try:
            think_idx = token_strs.index("</think>")
        except ValueError:
            think_idx = len(token_strs)

        think_entropys = entropys[:think_idx]
        if think_entropys.numel() > 1:
            diffs = think_entropys[:-1] - think_entropys[1:]
            valid_reliefs = F.relu(diffs - epsilon_err)
            total_relief = valid_reliefs.sum().item()
            T = think_entropys.numel()
            length_normalization = torch.log(torch.tensor(T + 1.0)).item()
            
            if length_normalization > 0:
                relief_reward = lambda_err * (total_relief / length_normalization)

    # Core modification 1: Package base score and exploration score independently with a hard cap of 1.5
    task_score = min(base_score + relief_reward, 1.5)

    # === 2. RRER Length Modifier ===

    alpha = 0.3    # Multiplicative coefficient, directly determines the max reward/penalty magnitude

    gamma = 0.5    
    epsilon_std = 1e-5
    
    seq_len = len(valid_response_ids)
    
    if group_lengths and len(group_lengths) > 1:
        lengths_arr = np.array(group_lengths)
        mu_L = np.mean(lengths_arr)
        sigma_L = np.std(lengths_arr)
        
        # Handle edge case where all group lengths are extremely close (e.g., early collapse)
        if sigma_L < epsilon_std:
            lambda_val = 0.0
        else:
            # 计算 Z-score 并在 [-1, 1] 内进行 Tanh 平滑映射
            z_i = (seq_len - mu_L) / (sigma_L + epsilon_std)
            lambda_val = math.tanh(-gamma * z_i) 
    else:
        # Fallback: if no intra-group comparison available, use absolute thresholds
        # (assume 2048 as midpoint, 1024 as std width). Apply Tanh here to ensure output stays in [-1, 1]
        z_i = (seq_len - 2048.0) / 1024.0
        lambda_val = math.tanh(-gamma * z_i)

    # Core modification 2: Asymmetric clipping to prevent model from exploiting length rewards with wrong answers
    if is_correct:
        f_val = lambda_val            # Correct: reward short, penalize long
    else:
        f_val = min(0.0, lambda_val)  # Incorrect: penalize long, no reward for short

    len_reward = alpha * f_val

    final_score = task_score 
    # final_score = base_score + len_reward    # Use this during Phase 2
    return max(final_score, -0.5)