# ERR+: Sequential Entropy Resolution for Efficient and Decisive LLM Reasoning

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official implementation of the paper **"ERR+: Sequential Entropy Resolution for Efficient and Decisive LLM Reasoning"**.

> **TL;DR** — Correct reasoning traces exhibit more frequent and larger token-level entropy drops than incorrect ones. ERR+ translates this into a two-phase RLVR framework: Phase 1 rewards entropy resolution in the thinking phase (ERR), and Phase 2 applies a difficulty-aware length signal via peer-group comparison (RRER). The result: **higher accuracy and shorter responses across model families and benchmarks**, without trading one for the other.

## Overview

Large reasoning models (e.g., DeepSeek-R1, Qwen3) achieve strong performance by generating extended chain-of-thought traces and training with reinforcement learning from verifiable rewards (RLVR). However, binary correctness rewards provide no signal about reasoning *quality* — a concise, decisive trace and a verbose, meandering trace receive identical rewards if both reach the right answer.

### Phase 1: Entropy Relief Reward (ERR)

We observe empirically that **correct reasoning traces exhibit more frequent and deeper token-level entropy drops** within the `<think>` phase than incorrect ones. ERR rewards the *resolution* of uncertainty — not the suppression of entropy itself. This leaves exploratory high-entropy states unconstrained while incentivizing decisive commitment at reasoning milestones.

$$\text{ERR}(y) = \frac{\sum_{t=2}^{T_k} \max(H_{t-1} - H_t - \epsilon, 0)}{\log(T_k + 1)}$$

where $H_t$ is the token-level entropy at thinking position $t$, $\epsilon$ suppresses noise, and log-normalization by thinking length $T_k$ prevents the model from generating longer traces purely to accumulate more drops.

### Phase 2: Robust Relative Efficiency Reward (RRER)

Once correct reasoning structure is established, Phase 2 refines the model with a difficulty-aware length signal. Instead of applying uniform length pressure, RRER scores each response against co-generated peers via a **tanh-transformed within-group z-score**:

$$z_i = \frac{L_i - \mu_L}{\sigma_L + \varepsilon}, \quad \lambda_i = \tanh(-\gamma z_i)$$

Using the within-group mean $\mu_L$ as an implicit difficulty proxy (harder problems naturally elicit longer responses), RRER applies asymmetric allocation: correct responses receive bonuses for conciseness, while incorrect responses are only penalized for excessive length.

### Why Two Phases?

We provide a formal gradient-conflict analysis (Theorem 1) showing that joint optimization of ERR and RRER produces conflicting policy gradient directions in early training. The mid-exploration tokens that RRER would prune are precisely the high-entropy context preceding committed drops — removing them directly harms ERR. After Phase 1 converges, this conflict vanishes, making the sequential design provably preferable.

## Key Results

| Model | Method | GSM8K Acc | Avg Tokens | Avg Acc (5 benchmarks) | Avg Tokens |
|-------|--------|-----------|------------|------------------------|------------|
| DS-R1-1.5B | Base | 84.6 | 2076 | 63.0 | 6818 |
| | GRPO | 86.6 | 2281 | 64.2 | 7240 |
| | PEAR | 85.3 | 1949 | 63.6 | 6621 |
| | **ERR+** | **88.6** | **1324** | **68.4** | **5450** |
| Qwen3-8B | Base | 93.9 | 2226 | 79.2 | 5012 |
| | GRPO | 94.6 | 2102 | 79.4 | 5509 |
| | PEAR | 93.8 | 1639 | 80.0 | 4360 |
| | **ERR+** | **96.0** | **1798** | **82.5** | **4495** |

On DeepSeek-R1-Distill-Qwen-1.5B, ERR+ achieves **+8.6% accuracy** while **reducing response length by 20.1%** compared to the base model. Unlike PEAR (which compresses length but degrades accuracy) and GRPO (which improves accuracy without shortening responses), ERR+ is the only method that reliably improves both metrics.

Evaluated across: GSM8K, AIME 2024, AMC23, MATH-500, MMLU-STEM.

## Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU(s)
- PyTorch 2.6

### Setup

```bash
conda create -n verl python=3.12
conda activate verl

# Install dependencies
pip install -r requirements.txt

# Install flash-attention (adjust wheel name for your CUDA/PyTorch version)
pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

# Verify installation
python -c "import verl; print('VERL installed successfully')"
```

## Quick Start

### Training on GSM8K

```bash
bash run_qwen3-8b.sh
```

This launches GRPO training with the ERR+ reward function on the GSM8K dataset using DeepSeek-R1-Distill-Qwen-1.5B.

### Using the ERR+ Reward Function

The core reward logic is in `verl/utils/reward_score/err.py`. To use it as a custom reward function with the verl framework:

```python
# In your training config, specify:
# reward_model.custom_reward_function.path=verl/utils/reward_score/err.py
# reward_model.custom_reward_function.name=compute_score
```

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon_err` | 0.01 | Noise threshold for entropy drops |
| `lambda_err` | 0.3 | Weight of ERR bonus (Phase 1) |
| `alpha` | 0.3 | Weight of RRER length modifier (Phase 2) |
| `gamma` | 0.5 | Tanh sensitivity for z-score scaling |
| `R_max` | 1.5 | Hard cap on total Phase-1 reward |

### Phased Training Schedule

1. **Phase 1** (ERR): Train with entropy relief reward. The reward function gated on correctness — only correct responses receive the entropy-drop bonus. Use `final_score = task_score` (base + ERR bonus, capped at 1.5).

2. **Phase 2** (RRER): Switch to `final_score = base_score + len_reward` for length refinement. The within-group relative scoring automatically adapts to problem difficulty.

## Repository Structure

```
err_response/
├── verl/                          # Core VERL framework (based on verl by Bytedance)
│   ├── trainer/
│   │   ├── main_ppo.py            # Main PPO training entry point
│   │   └── ppo/
│   │       ├── reward.py          # Reward function loading & management
│   │       └── ray_trainer.py     # Ray-based distributed PPO trainer
│   ├── utils/
│   │   └── reward_score/
│   │       ├── err.py             # ★ ERR+ reward implementation
│   │       ├── gsm8k.py           # GSM8K evaluation utilities
│   │       └── ...
│   └── workers/
│       ├── reward_manager/        # Reward manager implementations (naive, batch, dapo, etc.)
│       └── rollout/               # Rollout workers (vLLM, SGLang)
├── run_qwen3-8b.sh                # Example training script
├── data/                          # Dataset preprocessing scripts
├── examples/                      # Additional training examples
├── scripts/                       # Utility scripts
└── requirements.txt               # Python dependencies
```


## License

This project is built on [verl](https://github.com/volcengine/verl) by Bytedance and is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgements

- [verl](https://github.com/volcengine/verl) — Volcano Engine Reinforcement Learning for LLMs
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) — The base reasoning models used in our experiments
- [Qwen3](https://github.com/QwenLM/Qwen3) — Qwen3 model family
