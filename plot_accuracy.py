"""
对比两个模型在 GSM8K 数学题上的准确率。

模型:
  1. DeepSeek-R1-Distill-Qwen-1.5B (原始蒸馏模型)
  2. r1-1.5b-base3 (微调后模型)

方法: 对每道题用 chat template 构造 prompt, 模型 greedy decode 生成回答,
      用正则 "#### 数字" 提取答案并与 ground truth 比较。
"""

import os
import re
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ============ 配置 ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data/gsm8k/socratic/train-00000-of-00001.parquet")
MODEL_PATHS = {
    "DeepSeek-R1-Distill-Qwen-1.5B": os.path.join(BASE_DIR, "model/DeepSeek-R1-Distill-Qwen-1.5B"),
    "r1-1.5b-base3": os.path.join(BASE_DIR, "model/r1-1.5b-base3"),
}
NUM_SAMPLES = 100       # 取前 N 道题
MAX_NEW_TOKENS = 2048
OUTPUT_DIR = os.path.join(BASE_DIR, "entropy_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_SOLUTION_CLIP_CHARS = 300


def extract_ground_truth(answer_str):
    """从 GSM8K 原始 answer 字段中提取 #### 后的数字作为 ground truth"""
    solutions = re.findall(r"####\s*([\-\d\.\,]+)", answer_str)
    if solutions:
        return solutions[-1].replace(",", "").strip()
    return None


def extract_solution(solution_str, method="strict"):
    """从模型生成文本中提取答案"""
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        solutions = re.findall(r"####\s*([\-\d\.\,]+)", solution_str)
        if not solutions:
            return None
        return solutions[-1].replace(",", "").replace("$", "")
    else:  # flexible
        answer = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        for final_answer in reversed(answer):
            if final_answer not in ["", "."]:
                return final_answer
        return None


def load_data(data_path, num_samples):
    """加载 GSM8K 数据，返回 questions(messages格式) 和 ground_truths"""
    df = pd.read_parquet(data_path)
    questions = []
    ground_truths = []
    for i in range(min(num_samples, len(df))):
        row = df.iloc[i]
        q = row["question"]
        messages = [
            {"role": "user", "content": q + "\nLet's think step by step and output the final answer after \"####\"."}
        ]
        questions.append(messages)
        gt = extract_ground_truth(row["answer"])
        ground_truths.append(gt)
    return questions, ground_truths


def generate_answer(model, tokenizer, messages, max_new_tokens, device):
    """Greedy decode 生成回答，使用 KV cache 加速"""
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    eos_token_id = tokenizer.eos_token_id
    past_key_values = None
    current_input = input_ids
    generated_token_ids = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(current_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            generated_token_ids.append(next_token_id.item())
            if next_token_id.item() == eos_token_id:
                break
            current_input = next_token_id.unsqueeze(0).unsqueeze(0)

    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return generated_text


def evaluate_model(model_name, model_path, questions, ground_truths, device):
    """评估单个模型的准确率，返回逐题结果"""
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    correct_strict = 0
    correct_flexible = 0
    total = 0
    results = []  # (question_idx, gt, pred_strict, pred_flexible, is_correct_strict, is_correct_flexible)

    for i, (messages, gt) in enumerate(tqdm(
        zip(questions, ground_truths), total=len(questions),
        desc=f"Evaluating [{model_name}]"
    )):
        gen_text = generate_answer(model, tokenizer, messages, MAX_NEW_TOKENS, device)

        pred_strict = extract_solution(gen_text, method="strict")
        pred_flexible = extract_solution(gen_text, method="flexible")

        is_correct_strict = (pred_strict is not None and pred_strict == gt)
        is_correct_flexible = (pred_flexible is not None and pred_flexible == gt)

        if is_correct_strict:
            correct_strict += 1
        if is_correct_flexible:
            correct_flexible += 1
        total += 1

        results.append({
            "idx": i,
            "gt": gt,
            "pred_strict": pred_strict,
            "pred_flexible": pred_flexible,
            "correct_strict": is_correct_strict,
            "correct_flexible": is_correct_flexible,
        })

        if i < 3:
            print(f"\n--- Question {i+1} ---")
            print(f"Q: {messages[0]['content'][:80]}...")
            print(f"Generated: {gen_text[:200]}...")
            print(f"GT: {gt} | Pred(strict): {pred_strict} | Pred(flexible): {pred_flexible}")
            print(f"Correct(strict): {is_correct_strict} | Correct(flexible): {is_correct_flexible}")

    # 清理显存
    del model
    torch.cuda.empty_cache()

    acc_strict = correct_strict / total * 100 if total > 0 else 0
    acc_flexible = correct_flexible / total * 100 if total > 0 else 0

    print(f"\n{model_name} Results:")
    print(f"  Strict accuracy:   {correct_strict}/{total} = {acc_strict:.1f}%")
    print(f"  Flexible accuracy: {correct_flexible}/{total} = {acc_flexible:.1f}%")

    return {
        "results": results,
        "acc_strict": acc_strict,
        "acc_flexible": acc_flexible,
        "correct_strict": correct_strict,
        "correct_flexible": correct_flexible,
        "total": total,
    }


def plot_accuracy(all_results, output_dir):
    """绘制准确率对比图"""
    model_names = list(all_results.keys())
    colors = {"DeepSeek-R1-Distill-Qwen-1.5B": "#2196F3", "r1-1.5b-base3": "#FF5722"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- 图1: 柱状图对比（Flexible 匹配，因为模型不输出 #### 格式） ---
    ax1 = axes[0]
    x = np.arange(len(model_names))
    width = 0.45

    flexible_accs = [all_results[m]["acc_flexible"] for m in model_names]
    bar_colors = [colors[m] for m in model_names]

    bars = ax1.bar(x, flexible_accs, width, color=bar_colors, alpha=0.85,
                   edgecolor='white', linewidth=1.5)

    # 在柱子上标注数值
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 1.0, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax1.set_ylabel("Accuracy (%)", fontsize=13)
    ax1.set_title(f"GSM8K Accuracy Comparison\n({all_results[model_names[0]]['total']} samples, greedy decoding)",
                  fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, fontsize=11, ha='center')
    ax1.set_ylim(0, max(flexible_accs) + 15)
    ax1.grid(axis='y', alpha=0.3)

    # --- 图2: 累计准确率曲线（随样本数增加） ---
    ax2 = axes[1]
    for model_name in model_names:
        results = all_results[model_name]["results"]
        cumulative_correct = np.cumsum([r["correct_flexible"] for r in results])
        cumulative_total = np.arange(1, len(results) + 1)
        cumulative_acc = cumulative_correct / cumulative_total * 100
        ax2.plot(cumulative_total, cumulative_acc, label=model_name,
                color=colors[model_name], linewidth=2)

    ax2.set_xlabel("Number of Questions", fontsize=13)
    ax2.set_ylabel("Cumulative Accuracy (%)", fontsize=13)
    ax2.set_title("Cumulative Accuracy Over Questions", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(all_results[model_names[0]]["results"]))
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nAccuracy plot saved to: {save_path}")


def main():
    print("=" * 60)
    print("Accuracy Comparison: GSM8K")
    print("=" * 60)

    questions, ground_truths = load_data(DATA_PATH, NUM_SAMPLES)
    print(f"Loaded {len(questions)} questions")

    all_results = {}
    devices = ["cuda:0", "cuda:1"]

    for idx, (model_name, model_path) in enumerate(MODEL_PATHS.items()):
        device = devices[idx % len(devices)]
        result = evaluate_model(model_name, model_path, questions, ground_truths, device)
        all_results[model_name] = result

    # 绘图
    plot_accuracy(all_results, OUTPUT_DIR)

    # 打印总结
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)
    for model_name, data in all_results.items():
        print(f"\n{model_name}:")
        print(f"  Strict:   {data['correct_strict']}/{data['total']} = {data['acc_strict']:.1f}%")
        print(f"  Flexible: {data['correct_flexible']}/{data['total']} = {data['acc_flexible']:.1f}%")


if __name__ == "__main__":
    main()
