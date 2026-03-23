"""
对比两个模型在 GSM8K 测试集中特定 case 上，生成过程中逐 token 的熵变化曲线。
横坐标为绝对 token 位置。每道题单独一张子图，同图中两个模型叠加对比。
"""

import os
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
DATA_PATH = os.path.join(BASE_DIR, "data/gsm8k/socratic/test-00000-of-00001.parquet")
MODEL_PATHS = {
    "DeepSeek-R1-Distill-Qwen-1.5B": os.path.join(BASE_DIR, "model/DeepSeek-R1-Distill-Qwen-1.5B"),
    "r1-1.5b-base3": os.path.join(BASE_DIR, "model/r1-1.5b-base3"),
}
# 选取的 case 索引（测试集中的行号）
CASE_INDICES = [0, 5, 10, 20, 50]
MAX_NEW_TOKENS = 2048
OUTPUT_DIR = os.path.join(BASE_DIR, "entropy_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {"DeepSeek-R1-Distill-Qwen-1.5B": "#2196F3", "r1-1.5b-base3": "#FF5722"}


def compute_entropy_per_token(logits):
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.item()


def generate_with_entropy(model, tokenizer, messages, max_new_tokens, device):
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    entropies = []
    generated_token_ids = []
    eos_token_id = tokenizer.eos_token_id
    past_key_values = None
    current_input = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(current_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[0, -1, :]
            ent = compute_entropy_per_token(next_token_logits)
            entropies.append(ent)
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            generated_token_ids.append(next_token_id.item())
            if next_token_id.item() == eos_token_id:
                break
            current_input = next_token_id.unsqueeze(0).unsqueeze(0)

    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return entropies, generated_text


def main():
    print("=" * 60)
    print("Per-Case Entropy Comparison (GSM8K Test Set)")
    print("=" * 60)

    # 加载测试集
    df = pd.read_parquet(DATA_PATH)
    print(f"Test set: {len(df)} questions")

    # 构造选中 case 的 messages
    cases = []
    for idx in CASE_INDICES:
        if idx >= len(df):
            print(f"Warning: index {idx} out of range, skipping")
            continue
        q = df.iloc[idx]["question"]
        messages = [
            {"role": "user", "content": q + "\nLet's think step by step and output the final answer after \"####\"."}
        ]
        cases.append({"idx": idx, "question": q, "messages": messages})

    num_cases = len(cases)
    print(f"Selected {num_cases} cases: indices {[c['idx'] for c in cases]}")

    # 对每个模型跑所有 case，收集结果
    # results[model_name][case_i] = {"entropies": [...], "gen_text": "..."}
    all_results = {}
    devices = ["cuda:0", "cuda:1"]

    for model_idx, (model_name, model_path) in enumerate(MODEL_PATHS.items()):
        device = devices[model_idx % len(devices)]
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name} on {device}")
        print(f"{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            device_map=device, trust_remote_code=True,
        )
        model.eval()

        model_results = []
        for ci, case in enumerate(tqdm(cases, desc=f"[{model_name}]")):
            entropies, gen_text = generate_with_entropy(
                model, tokenizer, case["messages"], MAX_NEW_TOKENS, device
            )
            model_results.append({"entropies": entropies, "gen_text": gen_text})
            print(f"  Case {case['idx']}: {len(entropies)} tokens, "
                  f"mean entropy={np.mean(entropies):.3f}")

        all_results[model_name] = model_results
        del model
        torch.cuda.empty_cache()

    # ============ 绘图 ============
    model_names = list(MODEL_PATHS.keys())

    # --- 图1: 每个 case 一个子图，两个模型叠加 ---
    ncols = min(3, num_cases)
    nrows = (num_cases + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    if num_cases == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ci, case in enumerate(cases):
        ax = axes[ci]
        for model_name in model_names:
            ent = all_results[model_name][ci]["entropies"]
            tokens = np.arange(len(ent))
            # 轻微平滑便于观察趋势
            window = max(1, len(ent) // 80)
            if window > 1:
                smoothed = np.convolve(ent, np.ones(window)/window, mode='same')
            else:
                smoothed = np.array(ent)
            ax.plot(tokens, smoothed, label=f"{model_name} ({len(ent)} tok)",
                    color=COLORS[model_name], linewidth=1.2, alpha=0.85)
            # 半透明原始值
            ax.plot(tokens, ent, color=COLORS[model_name], linewidth=0.3, alpha=0.25)

        q_short = case["question"][:60].replace("\n", " ")
        ax.set_title(f"Case #{case['idx']}: \"{q_short}...\"", fontsize=10)
        ax.set_xlabel("Token Position", fontsize=11)
        ax.set_ylabel("Entropy (nats)", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # 隐藏多余子图
    for ci in range(num_cases, len(axes)):
        axes[ci].set_visible(False)

    fig.suptitle("Per-Case Token-Level Entropy: Model Comparison (GSM8K Test Set)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "entropy_cases.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nPer-case plot saved to: {save_path}")

    # --- 图2: 所有 case 叠加在一张大图上（分模型两个子图） ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))
    case_colors = plt.cm.tab10(np.linspace(0, 1, num_cases))

    for mi, model_name in enumerate(model_names):
        ax = axes2[mi]
        for ci, case in enumerate(cases):
            ent = all_results[model_name][ci]["entropies"]
            tokens = np.arange(len(ent))
            window = max(1, len(ent) // 60)
            if window > 1:
                smoothed = np.convolve(ent, np.ones(window)/window, mode='same')
            else:
                smoothed = np.array(ent)
            ax.plot(tokens, smoothed, label=f"Case #{case['idx']}",
                    color=case_colors[ci], linewidth=1.5, alpha=0.8)
        ax.set_title(model_name, fontsize=13)
        ax.set_xlabel("Token Position", fontsize=12)
        ax.set_ylabel("Entropy (nats)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    fig2.suptitle("Entropy Curves by Case (GSM8K Test Set)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path2 = os.path.join(OUTPUT_DIR, "entropy_cases_by_model.png")
    plt.savefig(save_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"By-model plot saved to: {save_path2}")

    # 打印详细信息
    print("\n" + "=" * 60)
    print("Case Details")
    print("=" * 60)
    for ci, case in enumerate(cases):
        print(f"\n--- Case #{case['idx']} ---")
        print(f"Q: {case['question'][:120]}...")
        for model_name in model_names:
            r = all_results[model_name][ci]
            print(f"  [{model_name}] {len(r['entropies'])} tokens, "
                  f"mean={np.mean(r['entropies']):.3f}, "
                  f"max={max(r['entropies']):.3f}, "
                  f"min={min(r['entropies']):.3f}")
            print(f"    Answer: {r['gen_text'][:100]}...")


if __name__ == "__main__":
    main()
