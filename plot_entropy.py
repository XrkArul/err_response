"""
对比两个模型在 GSM8K 数学题上生成过程中的 token-level 熵(entropy)变化曲线。

模型:
  1. DeepSeek-R1-Distill-Qwen-1.5B (原始蒸馏模型)
  2. r1-1.5b-base3 (微调后模型)

方法: 对每道题,用 chat template 构造 prompt, 模型 greedy decode 生成回答,
      在每一步记录输出分布的熵 H = -sum(p * log(p)),最后对所有题目取平均。
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
DATA_PATH = os.path.join(BASE_DIR, "data/gsm8k/socratic/train-00000-of-00001.parquet")
MODEL_PATHS = {
    "DeepSeek-R1-Distill-Qwen-1.5B": os.path.join(BASE_DIR, "model/DeepSeek-R1-Distill-Qwen-1.5B"),
    "r1-1.5b-base3": os.path.join(BASE_DIR, "model/r1-1.5b-base3"),
}
NUM_SAMPLES = 50        # 取前 N 道题（全量 7473 会很慢）
MAX_NEW_TOKENS = 2048   # 最大生成 token 数
OUTPUT_DIR = os.path.join(BASE_DIR, "entropy_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_questions(data_path, num_samples):
    """加载 GSM8K 数据，构造 chat messages 格式的 prompt"""
    df = pd.read_parquet(data_path)
    questions = []
    for i in range(min(num_samples, len(df))):
        q = df.iloc[i]["question"]
        messages = [
            {"role": "user", "content": q + "\nLet's think step by step and output the final answer after \"####\"."}
        ]
        questions.append(messages)
    return questions


def compute_entropy_per_token(logits):
    """
    给定 logits (vocab_size,)，计算该分布的熵。
    H = -sum(p * log(p))，其中 p = softmax(logits)
    """
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.item()


def generate_with_entropy(model, tokenizer, messages, max_new_tokens, device):
    """
    对给定的 prompt（messages 格式）进行 greedy decode 生成，
    逐 token 记录输出分布的熵。使用 KV cache 加速。
    
    Returns:
        entropies: list of float, 每个生成 token 对应的熵
        generated_text: str, 生成的完整文本
    """
    # 用 chat template 构造输入
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
        for step in range(max_new_tokens):
            outputs = model(current_input, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            
            # 取最后一个 token 的 logits
            next_token_logits = outputs.logits[0, -1, :]
            
            # 计算熵
            ent = compute_entropy_per_token(next_token_logits)
            entropies.append(ent)
            
            # Greedy decode: 取 argmax
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            generated_token_ids.append(next_token_id.item())
            
            # 检查是否生成了 EOS
            if next_token_id.item() == eos_token_id:
                break
            
            # 下一步只输入新 token（KV cache 已有历史）
            current_input = next_token_id.unsqueeze(0).unsqueeze(0)
    
    # 解码生成部分
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    
    return entropies, generated_text


def run_model(model_name, model_path, questions, device):
    """对一个模型运行所有题目, 收集每题的 entropy 列表"""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"Path: {model_path}")
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
    
    all_entropies = []  # list of lists
    
    for i, messages in enumerate(tqdm(questions, desc=f"Generating [{model_name}]")):
        entropies, gen_text = generate_with_entropy(
            model, tokenizer, messages, MAX_NEW_TOKENS, device
        )
        all_entropies.append(entropies)
        
        if i < 2:  # 打印前两道题的生成结果
            print(f"\n--- Question {i+1} ---")
            print(f"Q: {messages[0]['content'][:100]}...")
            print(f"Generated ({len(entropies)} tokens): {gen_text[:200]}...")
            print(f"Entropy range: [{min(entropies):.3f}, {max(entropies):.3f}], mean: {np.mean(entropies):.3f}")
    
    # 清理显存
    del model
    torch.cuda.empty_cache()
    
    return all_entropies


def align_and_average(all_entropies):
    """
    将不同长度的 entropy 序列对齐后取平均。
    方法: 将每条序列归一化到 [0, 1] 的相对位置，然后插值到统一的 N 个点上。
    同时也计算 token 位置的绝对平均（按最大长度 pad）。
    """
    if not all_entropies:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # --- 方法1: 归一化位置插值 ---
    num_interp_points = 500
    interp_x = np.linspace(0, 1, num_interp_points)
    interp_entropies = []
    
    for ent_list in all_entropies:
        if len(ent_list) < 2:
            continue
        x = np.linspace(0, 1, len(ent_list))
        y = np.array(ent_list)
        interp_y = np.interp(interp_x, x, y)
        interp_entropies.append(interp_y)
    
    if interp_entropies:
        interp_entropies = np.array(interp_entropies)
        mean_interp = np.mean(interp_entropies, axis=0)
        std_interp = np.std(interp_entropies, axis=0)
    else:
        mean_interp = np.zeros(num_interp_points)
        std_interp = np.zeros(num_interp_points)
    
    # --- 方法2: 绝对 token 位置 (截断到中位长度) ---
    lengths = [len(e) for e in all_entropies]
    median_len = int(np.median(lengths))
    abs_entropies = []
    for ent_list in all_entropies:
        arr = np.array(ent_list[:median_len])
        if len(arr) < median_len:
            arr = np.pad(arr, (0, median_len - len(arr)), constant_values=np.nan)
        abs_entropies.append(arr)
    abs_entropies = np.array(abs_entropies)
    mean_abs = np.nanmean(abs_entropies, axis=0)
    std_abs = np.nanstd(abs_entropies, axis=0)
    
    return interp_x, mean_interp, std_interp, mean_abs, std_abs, median_len


def plot_results(results, output_dir):
    """绘制对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    
    colors = {"DeepSeek-R1-Distill-Qwen-1.5B": "#2196F3", "r1-1.5b-base3": "#FF5722"}
    
    # --- 图1: 归一化位置的熵曲线 ---
    ax1 = axes[0]
    for model_name, data in results.items():
        interp_x, mean_interp, std_interp = data["interp_x"], data["mean_interp"], data["std_interp"]
        color = colors[model_name]
        ax1.plot(interp_x * 100, mean_interp, label=model_name, color=color, linewidth=1.5)
        ax1.fill_between(interp_x * 100, mean_interp - std_interp, mean_interp + std_interp,
                         alpha=0.15, color=color)
    
    ax1.set_xlabel("Relative Generation Progress (%)", fontsize=13)
    ax1.set_ylabel("Entropy (nats)", fontsize=13)
    ax1.set_title("Token-level Entropy During Generation\n(Normalized Position)", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    
    # --- 图2: 绝对 token 位置的熵曲线 ---
    ax2 = axes[1]
    for model_name, data in results.items():
        mean_abs, std_abs, median_len = data["mean_abs"], data["std_abs"], data["median_len"]
        x = np.arange(len(mean_abs))
        color = colors[model_name]
        ax2.plot(x, mean_abs, label=model_name, color=color, linewidth=1.5)
        ax2.fill_between(x, mean_abs - std_abs, mean_abs + std_abs,
                         alpha=0.15, color=color)
    
    ax2.set_xlabel("Token Position (absolute)", fontsize=13)
    ax2.set_ylabel("Entropy (nats)", fontsize=13)
    ax2.set_title("Token-level Entropy During Generation\n(Absolute Token Position)", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "entropy_comparison.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {save_path}")
    
    # --- 额外: 单独画一张带 think 标记的图 ---
    fig2, ax3 = plt.subplots(figsize=(14, 6))
    for model_name, data in results.items():
        interp_x, mean_interp = data["interp_x"], data["mean_interp"]
        color = colors[model_name]
        # 用 rolling mean 平滑
        window = 10
        smoothed = np.convolve(mean_interp, np.ones(window)/window, mode='same')
        ax3.plot(interp_x * 100, smoothed, label=model_name, color=color, linewidth=2)
    
    ax3.set_xlabel("Relative Generation Progress (%)", fontsize=13)
    ax3.set_ylabel("Entropy (nats, smoothed)", fontsize=13)
    ax3.set_title("Smoothed Entropy Comparison: Thinking → Answering", fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)
    
    save_path2 = os.path.join(output_dir, "entropy_comparison_smoothed.png")
    plt.savefig(save_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Smoothed plot saved to: {save_path2}")


def main():
    print("=" * 60)
    print("Entropy Comparison: Model Generation on GSM8K")
    print("=" * 60)
    
    # 加载数据
    questions = load_questions(DATA_PATH, NUM_SAMPLES)
    print(f"Loaded {len(questions)} questions from {DATA_PATH}")
    
    results = {}
    
    # 两个模型分别在不同 GPU 上运行（顺序执行，节省显存）
    devices = ["cuda:0", "cuda:1"]
    
    for idx, (model_name, model_path) in enumerate(MODEL_PATHS.items()):
        device = devices[idx % len(devices)]
        all_entropies = run_model(model_name, model_path, questions, device)
        
        interp_x, mean_interp, std_interp, mean_abs, std_abs, median_len = align_and_average(all_entropies)
        
        results[model_name] = {
            "all_entropies": all_entropies,
            "interp_x": interp_x,
            "mean_interp": mean_interp,
            "std_interp": std_interp,
            "mean_abs": mean_abs,
            "std_abs": std_abs,
            "median_len": median_len,
        }
        
        # 保存原始数据
        save_path = os.path.join(OUTPUT_DIR, f"{model_name.replace('/', '_')}_entropies.npz")
        np.savez(save_path,
                 interp_x=interp_x, mean_interp=mean_interp, std_interp=std_interp,
                 mean_abs=mean_abs, std_abs=std_abs,
                 lengths=[len(e) for e in all_entropies])
        print(f"Raw data saved to: {save_path}")
    
    # 绘图
    plot_results(results, OUTPUT_DIR)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for model_name, data in results.items():
        lengths = [len(e) for e in data["all_entropies"]]
        all_ent = [e for ent_list in data["all_entropies"] for e in ent_list]
        print(f"\n{model_name}:")
        print(f"  Avg generation length: {np.mean(lengths):.1f} tokens")
        print(f"  Median generation length: {np.median(lengths):.1f} tokens")
        print(f"  Overall mean entropy: {np.mean(all_ent):.4f} nats")
        print(f"  Overall std entropy: {np.std(all_ent):.4f} nats")


if __name__ == "__main__":
    main()
