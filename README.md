# Annotating Indian Regional Biases using Large Language Models: Evaluation and Analysis

> Comprehensive evaluation of open-source LLMs for automatically annotating Indian regional biases using the IndRegBias dataset.

## Overview

Regional Bias is a form of social bias rooted in generalised assumptions about an individual/state/region based on their geographical identity, which is associated with its language, culture, socio-economic development, and political paradigm. This project evaluates open-source large language models on their ability to annotate the social comments as regional bias (RB) or non-regional bias (NRB), where the comments belong to three text-writing styles: **English**, **Code-Mixed**, and **Transliterated**.

We assess LLMs in two settings:
1. **Zero-shot prompting** — direct inference using chain-of-thought reasoning
2. **Fine-tuning with LoRA** — parameter-efficient supervised fine-tuning on 50% of IndRegBias, tested on the remaining 50% and 500 newly collected YouTube comments

### Key Findings

- All LLMs show **low agreement** (κ < 0.49) in zero-shot settings
- Fine-tuning yields **64–100%+ improvement** in annotation agreement (Cohen's κ)
- **Mixtral-MoE** achieves the best fine-tuned agreement on IndRegBias
- **Krutrim-2** (Indic LLM) achieves the highest κ (0.695) on newly collected set of YouTube comments
- English comments are easiest to annotate; code-mixed and transliterated texts remain challenging

---

## Models Evaluated

| Full Model Name | Short Name | Parameters |
|---|---|---|
| Qwen_3_8b | Qwen-8B | 8B |
| Qwen_3_32b | Qwen-32B | 32B |
| DeepSeek-R1-Distill-Qwen-14B | DSeek-14B | 14B |
| Krutrim-2-instruct | Krutrim-2 | 12B |
| Mixtral-8x7B-Instruct-v0.1 | Mixtral-MoE | 8×7B |
| LLaMa_3.1_8b | Llama-8B | 8B |
| Mistral_Nemo_12b | MNemo-12B | 12B |
| Mistral_7b_v0.3 | Mistral-7B | 7B |
| Gemma_3_27b_it | Gemma-27B | 27B |

---

## Dataset

We use the [IndRegBias](https://arxiv.org/abs/2601.06477) dataset consisting of 25,000 social media comments (13,015 RB / 11,985 NRB) from YouTube and Reddit.

| Category | RB | NRB | Total |
|---|---|---|---|
| Code-Mixed | 3,405 | 2,910 | 6,315 |
| Transliterated | 108 | 355 | 463 |
| English | 9,502 | 8,720 | 18,222 |
| **Total** | **13,015** | **11,985** | **25,000** |

Additionally, **500 newly collected YouTube comments** (balanced RB/NRB) are used for out-of-distribution evaluation.

---
## Project Structure

The codebase is organized into two self-contained production execution modules sitting directly at the repository root. Shared parameter definitions and logical parsing utility functions are imported directly within each folder from their respective local files:

```text
Annotating-Indian-Regional-Biases-using-Large-Language-Models-Evaluation-and-Analysis/
├── data/
|   ├── Dataset_final.csv       # 25k Master training dataset (gitignored)
│   └── sample_500.csv          # Target 500-comment out-of-distribution evaluation data
│
├── weights/
│   └── {model_alias}_best/     # Automatically isolated production adapter checkpoints
│
├── Zero-Shot/                  # Pipeline 1: Context-Free Baseline Generation
│   ├── config.py               # Zero-shot specific model registries and constants
│   ├── utils.py                # Regex extraction, reasoning-trace stripping & CSV repair
│   └── run_inference.py        # Master zero-shot inference engine runner
│
├── Fine-Tune/                  # Pipeline 2: Parameter-Efficient Fine-Tuning (PEFT)
│   ├── config.py               # Fine-tuning parameters, LoRA ranks, and system prompts
│   ├── utils.py                # Environment initializers and chat template formatters
│   ├── train_cv.py             # 5-Fold Stratified CV engine (Saves absolute best fold)
│   └── inference_test.py       # Evaluation runner evaluating the 500-comment test sample
│
├── requirements.txt            # Operational environment dependencies
└── README.md                   # Repository documentation
```

##  Installation and Environment Setup

### 1. Clone and Install Dependencies
```bash
git clone [https://github.com/debby123p/Annotating-Indian-Regional-Biases-using-Large-Language-Models-Evaluation-and-Analysis.git](https://github.com/debby123p/Annotating-Indian-Regional-Biases-using-Large-Language-Models-Evaluation-and-Analysis.git)
cd Annotating-Indian-Regional-Biases-using-Large-Language-Models-Evaluation-and-Analysis
pip install -r requirements.txt
```

### 2. Data Access Note
The raw master training dataset (Dataset_final.csv) is not hosted in this public repository to protect data distribution rights. To replicate our benchmark findings, please request access directly from the paper's authors via the IndRegBias arXiv Link. Once acquired, create a local folder named data/ at the repository root and place your data files inside it.

### 3.Hardware Requirements

- **GPU**: NVIDIA H200 (~141 GB VRAM) or equivalent
- Fine-tuning uses 16-bit BFloat16 precision with LoRA to reduce memory requirements

---

## Usage

### 1. Zero-Shot Evaluation
To generate prompt-only predictions on a target dataset sample, change to the Zero-Shot directory and initialize the runtime inference pipeline wrapper:

```bash
cd Zero-Shot
python run_inference.py \
    --model qwen-8b \
    --data_path ../data/sample_500.csv \
    --output ../results/zero_shot/qwen_8b_baseline_results.csv \
    --gpu_id 0
```
To recover and pick up from where you left off after an unexpected server time-out or runtime disconnection, append the --resume flag parameter to skip pre-annotated index boundaries.

### 2. Fine-Tuning with LoRA
To run full parameter-efficient training splits over the 25k text comment training database, navigate to the Fine-Tune folder. The validation engine isolates variance across data splits and saves the optimal fold checkpoint weights to your /weights storage path tree:

```bash
cd Fine-Tune
python train_cv.py \
    --model mixtral-moe \
    --data_path ../data/Dataset_final.csv \
    --gpu_id 1
```

### 3. YouTube 500 Evaluation
Once optimal adapter boundaries have been isolated and written out by Phase 1, call the pure-inference execution script to test predictions over your custom out-of-distribution YouTube evaluations sample:

```bash
cd Fine-Tune
python inference_test.py \
    --model mixtral-moe \
    --test_path ../data/sample_500.csv \
    --gpu_id 1
```

---

## Fine-Tuning Configuration

| Parameter | Value |
|---|---|
| Method | LoRA (16-bit BFloat16) |
| LoRA Rank (r) | 16 |
| LoRA Alpha (α) | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | All linear layers (q, k, v, o, gate, up, down) |
| Epochs | 10 |
| Learning Rate | 2e-4 |
| LR Scheduler | Cosine (warmup ratio: 0.03) |
| Optimizer | AdamW (8-bit) |
| Batch Size | 8 (device) × 4 (accumulation) = 32 (effective) |
| Max Length | 2048 tokens |
| Early Stopping | Patience: 3, Threshold: 0.01 |
| Validation | 5-Fold Stratified Cross-Validation |
| Data Split | 40% Train / 10% Val / 50% Test |

---

## Evaluation Metric

We use **Cohen's Kappa (κ)** to measure annotation agreement:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where $p_o$ is observed agreement and $p_e$ is expected chance agreement.

---

## Citation

```bibtex
@article{panda2026indregbias,
  title={IndRegBias: A Dataset for Studying Indian Regional Biases in English and Code-Mixed Social Media Comments},
  author={Panda, Debasmita and Anil, Akash and Shukla, Neelesh Kumar},
  journal={arXiv preprint arXiv:2601.06477},
  year={2026}
}
```

---

## Acknowledgments

- [IndRegBias Dataset](https://arxiv.org/abs/2601.06477)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
