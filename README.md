# Annotating Indian Regional Biases using Large Language Models: Evaluation and Analysis

> Comprehensive evaluation of open-source LLMs for automatically annotating Indian regional biases using the IndRegBias dataset.

## Overview

Regional bias (RB) is a type of social bias originating from stereotypes toward particular geographical identities. This project evaluates **nine open-source LLMs** for annotating Indian regional biases across three text-writing styles: **English**, **Code-Mixed**, and **Transliterated**.

We assess LLMs in two settings:
1. **Zero-shot prompting** — direct inference using chain-of-thought reasoning
2. **Fine-tuning with LoRA** — parameter-efficient supervised fine-tuning on 50% of IndRegBias, tested on the remaining 50% and 500 newly collected YouTube comments

### Key Findings

- All LLMs show **low agreement** (κ < 0.49) in zero-shot settings
- Fine-tuning yields **64–100%+ improvement** in annotation agreement (Cohen's κ)
- **Mixtral-MoE** achieves the best fine-tuned agreement on IndRegBias
- **Krutrim-2** (Indic LLM) achieves the highest κ (0.695) on newly collected YouTube comments
- English comments are easiest to annotate; code-mixed and transliterated texts remain challenging

---

## Models Evaluated

| Full Model Name | Short Name | Parameters |
|---|---|---|
| Qwen_3_8b | Qwen-8B | 8B |
| Qwen_3_32b | Qwen-32B | 32B |
| DeepSeek-R1-Distill-Qwen-14B | DSeek-14B | 14B |
| Krutrim-2-instruct | Krutrim-2 | — |
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

```
IndRegBias-LLM-Annotation/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── .gitignore
│
├── configs/
│   ├── lora_config.yaml          # LoRA and fine-tuning hyperparameters
│   └── model_registry.yaml       # Model names, paths, and HF identifiers
│
├── prompts/
│   ├── zero_shot_prompt.txt      # Zero-shot annotation prompt
│   └── fine_tuning_template.txt  # Fine-tuning input template
│
├── src/
│   ├── zero_shot/
│   │   ├── __init__.py
│   │   └── inference.py          # Zero-shot annotation pipeline
│   │
│   ├── fine_tuning/
│   │   ├── __init__.py
│   │   ├── train.py              # LoRA fine-tuning script
│   │   └── data_preparation.py   # Stratified split generation
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Cohen's κ and agreement metrics
│   │   └── category_analysis.py  # Per-category (CM/Trans/Eng) evaluation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py        # Dataset loading and preprocessing
│       └── text_classifier.py    # Code-mixed / Transliterated / English classifier
│
├── scripts/
│   ├── run_zero_shot.sh          # Launch zero-shot evaluation
│   ├── run_fine_tuning.sh        # Launch fine-tuning pipeline
│   └── run_youtube_eval.sh       # Evaluate on YouTube 500
│
├── notebooks/
│   └── analysis.ipynb            # Result visualization and tables
│
├── data/
│   ├── raw/                      # Original IndRegBias data (not distributed)
│   ├── processed/                # Preprocessed and categorized data
│   ├── splits/                   # 5-fold stratified train/val/test splits
│   └── youtube_500/              # Newly collected YouTube comments
│
└── results/
    ├── zero_shot/                # Zero-shot κ scores per model
    ├── fine_tuning/              # Fine-tuning κ scores (5-fold)
    └── youtube/                  # YouTube 500 evaluation results
```

---

## Installation

```bash
git clone https://github.com/<your-username>/IndRegBias-LLM-Annotation.git
cd IndRegBias-LLM-Annotation
pip install -r requirements.txt
```

### Hardware Requirements

- **GPU**: NVIDIA H200 (~141 GB VRAM) or equivalent
- Fine-tuning uses 16-bit BFloat16 precision with LoRA to reduce memory requirements

---

## Usage

### 1. Zero-Shot Evaluation

```bash
bash scripts/run_zero_shot.sh --model qwen_3_8b --data_path data/raw/indregbias.csv
```

### 2. Fine-Tuning with LoRA

```bash
bash scripts/run_fine_tuning.sh \
  --model qwen_3_8b \
  --config configs/lora_config.yaml \
  --fold 1
```

### 3. YouTube 500 Evaluation

```bash
bash scripts/run_youtube_eval.sh \
  --model_checkpoint results/fine_tuning/qwen_3_8b/best_fold/ \
  --data_path data/youtube_500/
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

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [IndRegBias Dataset](https://arxiv.org/abs/2601.06477)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
