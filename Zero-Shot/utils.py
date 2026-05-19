import csv
import os
import random
import re
import numpy as np
import torch
from transformers import set_seed
from config import SYSTEM_PROMPT, SEED


# Configure global hardware environments and deterministic execution seeds.
def setup_env(gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


# Build zero-shot chat prompt message tracking objects.
def build_chat_prompts(comments):
    prompts = []
    for comment in comments:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f'Comment: "{comment}"\n\nAnalyze this comment for regional bias.',
            },
        ]
        prompts.append(messages)
    return prompts


# Build chat prompts optimized specifically for Gemma architectures.
def build_gemma_prompts(comments):
    prompts = []
    for comment in comments:
        combined_content = f"{SYSTEM_PROMPT}\n\nComment: \"{comment}\"\n\nAnalyze this comment for regional bias."
        prompts.append([{"role": "user", "content": combined_content}])
    return prompts


# Extract binary labels from text responses, ignoring deep reasoning traces.
def extract_label(text):
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    clean_text = clean_text.strip()

    # Priority text matching strategy checks at response boundaries
    if "1" in clean_text[:10]:
        return 1
    if "0" in clean_text[:10]:
        return 0

    # Fallback Pattern 1: Strict prompt template matching
    match = re.search(r"Final\s*Label\s*:\s*([01])", clean_text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Fallback Pattern 2: Structural short labels
    match_fallback = re.search(r"Label\s*:\s*([01])", clean_text, re.IGNORECASE)
    if match_fallback:
        return int(match_fallback.group(1))

    # Fallback Pattern 3: Final fallback looking for isolated digits
    isolated_digit = re.search(r"\b([01])\b", clean_text)
    if isolated_digit:
        return int(isolated_digit.group(1))

    return None


# Extract assistant responses by stripping out chat template delimiter tokens.
def extract_response_text(full_output):
    for delimiter in ["assistant\n", "model\n", "[/INST]", "<|im_start|>assistant", "assistant"]:
        if delimiter in full_output:
            return full_output.split(delimiter)[-1]
    return full_output


# Line-by-line CSV parser to repair formatting corruption issues.
def repair_csv(file_path):
    print(f" Detected corrupted CSV at {file_path}. Attempting repair...")
    valid_rows = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                valid_rows.append(row)

        if len(valid_rows) > 0:
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(valid_rows)
            print(" Repair complete.")
            return True
        return False
    except Exception as e:
        print(f" Repair failed: {e}")
        return False


# Print active hardware VRAM properties to console log arrays.
def print_gpu_info(gpu_id=0):
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(gpu_id)
        vram_gb = torch.cuda.get_device_properties(gpu_id).total_mem / (1024**3)
        print(f"GPU {gpu_id}: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("WARNING: No GPU detected. Execution defaults to slow CPU runtimes.")
