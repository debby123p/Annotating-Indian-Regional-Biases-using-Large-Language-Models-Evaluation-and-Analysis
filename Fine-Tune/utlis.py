import os
import re
import csv
import random
import numpy as np
import torch
from transformers import set_seed
from src.config import SYSTEM_PROMPT, SEED


# Configure global hardware runtimes and deterministic seeds.
def setup_env(gpu_id="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    set_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


# Build standardized chat template message dictionaries for training or inference.
#
# Args:
#    row     - Dict or Pandas series containing 'comment' and optionally 'level-1' keys
#    is_test - If True, skips appending the assistant gold-label role dictionary
def format_example(row, is_test=False):
    instruction = "Annotate the following comment as Regional Bias (1) or Non-Regional Bias (0)."
    
    # Combined user content optimal for strict context tracking in target architectures
    content = f"{SYSTEM_PROMPT}\n\n{instruction}\n\nComment: {row['comment']}"
    messages = [{"role": "user", "content": content}]
    
    if is_test:
        return {"messages": messages}
    else:
        messages.append({"role": "assistant", "content": str(int(row['level-1']))})
        return {"messages": messages}


# Extract the binary annotation from raw model generation sequences.
# Prioritizes text-boundary tokens before defaulting to regex fallbacks.
#
# Args:
#    text - Raw output string decoded directly from the model generation matrix
def extract_label(text):
    # Remove thinking tags or trace structures
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    clean_text = clean_text.strip()
    
    # Priority text matching check at boundaries
    if "1" in clean_text[:10]:
        return 1
    if "0" in clean_text[:10]:
        return 0
        
    # Standard priority match fallback: "Final Label: [0 or 1]"
    match = re.search(r"Final\s*Label\s*:\s*([01])", clean_text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Fallback match: "Label: [0 or 1]"
    match_fallback = re.search(r"Label\s*:\s*([01])", clean_text, re.IGNORECASE)
    if match_fallback:
        return int(match_fallback.group(1))

    # Catch-all isolated digit regex matching sequence
    isolated_digit = re.search(r"\b([01])\b", clean_text)
    if isolated_digit:
        return int(isolated_digit.group(1))

    return 0


# Scans a dataset file line by line, isolating and dropping corrupt rows
# to prevent Pandas ParserErrors during runtime loops.
#
# Args:
#    file_path - String or Path target configuration file location
def robust_repair_csv(file_path):
    print(f"Starting ROBUST repair on {file_path}...")
    valid_rows = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            try:
                for row in reader:
                    valid_rows.append(row)
            except csv.Error as e:
                print(f"   Found corruption. Stopping read. Error: {e}")

        if len(valid_rows) > 0:
            print(f"   Recovered {len(valid_rows)} valid rows. Overwriting file...")
            with open(file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(valid_rows)
            print("Repair complete.")
            return True
        else:
            print("Repair failed: No valid rows found.")
            return False
    except Exception as e:
        print(f"Fatal error during repair: {e}")
        return False


# Verify GPU resource availability and print execution matrix constraints.
#
# Args:
#    gpu_id - Integer device position mapping active hardware targets
def print_gpu_info(gpu_id=0):
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(gpu_id)
        vram_gb = torch.cuda.get_device_properties(gpu_id).total_mem / (1024**3)
        print(f"GPU {gpu_id}: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("WARNING: No GPU detected. Execution defaults to slow CPU array runtimes.")
