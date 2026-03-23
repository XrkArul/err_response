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
import torch
import torch.nn.functional as F

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
                 method='strict', format_score=0., score=1., tokenizer=None):

    old_log_probs, entropys = old_log_prob["old_log_probs"], old_log_prob["entropys"]

    relief_reward = 0.0
    
    # 参数设置 (Hyperparameters) - 论文实验部分可讨论这些参数的敏感性
    epsilon = 0.01   
    lambda_err = 0.3 # 奖励系数
    
    if entropys is not None and tokenizer is not None:
        token_strs = tokenizer.convert_ids_to_tokens(valid_response_ids)
        try:
            think_idx = token_strs.index("</think>")
        except ValueError:
            think_idx = len(token_strs)

        think_entropys = entropys[:think_idx]
        if think_entropys.numel() > 1:
            diffs = think_entropys[:-1] - think_entropys[1:]
            valid_reliefs = F.relu(diffs - epsilon)
            total_relief = valid_reliefs.sum().item()
            T = think_entropys.numel()
            length_normalization = torch.log(torch.tensor(T + 1.0)).item()
            
            # ERR = 总有效熵降 / log(长度)
            if length_normalization > 0:
                err_score = total_relief / length_normalization
                relief_reward = lambda_err * err_score

    # 提取答案
    answer = extract_solution(solution_str=solution_str, method=method)
    
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            # 核心设计：正确的基础分 + 效率加分
            # 这种设计比"扣分制"更稳定，鼓励模型去争取"高效率"的Bonus
            final_score = score + relief_reward
            
            # 设置上限，防止奖励爆炸
            return min(final_score, 1.5) 
        else:
            return format_score