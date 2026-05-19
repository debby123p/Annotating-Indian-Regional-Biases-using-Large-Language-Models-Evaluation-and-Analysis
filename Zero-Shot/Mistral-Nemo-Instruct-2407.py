import argparse
import csv
import os
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import cohen_kappa_score, classification_report


# Zero-Shot System Prompt
SYSTEM_PROMPT = (
    "You are an expert socio-linguist specializing in Indian regional biases. "
    "Your task is to analyze comments to detect if they contain "
    "'Regional Bias' (Label 1) or are 'Non-Biased' (Label 0).\n\n"
    "DEFINITIONS\n"
    "Label 1 (Regional Bias):\n"
    "The comment contains ANY of the following regarding an Indian state "
    "or community: Negative/Positive Stereotypes, Hate Speech/slurs, "
    "or Cultural Generalizations.\n\n"
    "Label 0 (Non-Regional Biased):\n"
    "Factual statements, non-related comments, or personal experiences "
    "without generalizing.\n\n"
    "OUTPUT FORMAT\n"
    "1. First, engage in a 'Thinking' process to analyze the sentiment.\n"
    "2. Then, output the final annotation strictly in this format:\n"
    "    Final Label: [0 or 1]"
)


# Build chat-formatted prompts for a batch of comments.
# Each prompt = system message (annotation instructions) + user message (comment).
# Returns list of message lists for tokenizer.apply_chat_template().
def build_chat_prompts(comments):
    prompts = []
    for comment in comments:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f'Comment: "{comment}"\n\n'
                    "Analyze this comment for regional bias."
                ),
            },
        ]
        prompts.append(messages)
    return prompts


# Extract predicted label from model output.
# Strips <think>...</think> reasoning traces (Qwen3, DeepSeek-R1 models)
# before searching for "Final Label: [0 or 1]".
# Returns 0 or 1 if found, None otherwise.
def extract_label(text):
    # Strip reasoning traces
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Primary pattern: "Final Label: 0" or "Final Label: 1"
    match = re.search(r"Final\s*Label\s*:\s*([01])", clean_text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Fallback pattern: "Label: 0" or "Label: 1"
    match_fallback = re.search(r"Label\s*:\s*([01])", clean_text, re.IGNORECASE)
    if match_fallback:
        return int(match_fallback.group(1))

    return None


# Extract assistant's response from full decoded output.
# Handles different chat format separators across Qwen, LLaMA, Mistral, etc.
def extract_response_text(full_output):
    for delimiter in ["assistant\n", "model\n", "[/INST]", "<|im_start|>assistant"]:
        if delimiter in full_output:
            return full_output.split(delimiter)[-1]
    return full_output


# Reads a corrupted file as raw text using csv reader, discards broken lines,
# and overwrites the target dataset to enable continuous pipeline parsing.
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


# Print GPU name and VRAM for verification.
def print_gpu_info(gpu_id=0):
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(gpu_id)
        vram_gb = torch.cuda.get_device_properties(gpu_id).total_mem / (1024**3)
        print(f"GPU {gpu_id}: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        print("WARNING: No GPU detected. Inference will be very slow.")


# Main zero-shot annotation function.
#
# Args:
#    model_name             - HuggingFace model ID (e.g., 'mistralai/Mistral-Nemo-Instruct-2407')
#    data_path              - Path to input CSV file with comments
#    output_path            - Path to save annotation results CSV
#    comment_col            - Column name containing comment text
#    label_col              - Column name containing ground truth labels
#    batch_size             - Number of comments per inference batch
#    max_new_tokens         - Maximum tokens to generate per comment
#    backup_every_n_batches - Save progress every N batches
#    resume                 - If True, resume from existing output file
def run_zero_shot(
    model_name,
    data_path,
    output_path,
    comment_col="comment",
    label_col="level-1",
    batch_size=8,
    max_new_tokens=512,
    backup_every_n_batches=10,
    resume=False,
):
    # Load original dataset
    print(f"Loading ORIGINAL dataset from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        print(f"Original dataset loaded: {len(df)} rows.")
    except Exception as e:
        print(f"Error loading original dataset: {e}")
        return

    # Initialize execution metrics tracking target parameters
    if "mistral_nemo_raw" not in df.columns:
        df["mistral_nemo_raw"] = None
    if "mistral_nemo_pred" not in df.columns:
        df["mistral_nemo_pred"] = None

    # Prepare output paths
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_path = output_dir / f"{Path(output_path).stem}_backup.csv"

    # Merge existing progress tracking updates
    if resume and os.path.exists(output_path):
        print(f"Found existing progress file: {output_path}")
        print("Merging existing results...")
        try:
            try:
                df_results = pd.read_csv(output_path)
            except pd.errors.ParserError:
                print("ParserError detected in results file. Repairing...")
                robust_repair_csv(output_path)
                df_results = pd.read_csv(output_path)

            if "mistral_nemo_pred" in df_results.columns:
                finished_indices = df_results[df_results["mistral_nemo_pred"].notna()].index
                df.loc[finished_indices, "mistral_nemo_pred"] = df_results.loc[finished_indices, "mistral_nemo_pred"]
                df.loc[finished_indices, "mistral_nemo_raw"] = df_results.loc[finished_indices, "mistral_nemo_raw"]
                print(f"Recovered {len(finished_indices)} completed predictions.")

        except Exception as e:
            print(f"Warning: Could not merge existing results. Starting fresh. Error: {e}")

    # Determine remaining pipeline step items batch boundaries
    remaining_indices = df[df["mistral_nemo_pred"].isna()].index
    if len(remaining_indices) > 0:
        start_index = remaining_indices[0]
        start_index = (start_index // batch_size) * batch_size
    else:
        start_index = len(df)

    print(f"Resuming at index: {start_index} of {len(df)}")
    if start_index >= len(df):
        print("Dataset is fully annotated!")
        return

    if label_col in df.columns:
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

    comments = df[comment_col].tolist()

    # Load model precision and target variables configurations
    print(f"Loading model: {model_name} (bfloat16)")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
    except Exception as e:
        print(f"Failed to load model. Error: {e}")
        return

    # Batched inference loop
    print(f"Annotating {len(comments)} comments (batch_size={batch_size})...")

    for i in tqdm(range(start_index, len(comments), batch_size)):
        batch_end = min(i + batch_size, len(comments))
        batch_comments = comments[i:batch_end]

        # Build chat prompts and apply model-specific chat template
        chat_prompts = build_chat_prompts(batch_comments)
        texts = [
            tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in chat_prompts
        ]

        # Tokenize and move to GPU
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        # Generate with sampling parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode and extract labels
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch_raw = []
        batch_preds = []

        for full_output in generated_texts:
            response_text = extract_response_text(full_output)
            pred_label = extract_label(response_text)

            batch_raw.append(response_text)
            batch_preds.append(pred_label)

        df.loc[i : batch_end - 1, "mistral_nemo_raw"] = batch_raw
        df.loc[i : batch_end - 1, "mistral_nemo_pred"] = batch_preds

        # Periodic backup to avoid data loss on crashes
        batch_num = (i - start_index) // batch_size
        if batch_num % backup_every_n_batches == 0 and batch_num > 0:
            df.to_csv(output_path, index=False)
            df.to_csv(backup_path, index=False)

    # Save final results
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Evaluation summary
    valid_df = df[df["mistral_nemo_pred"].notna()]
    if len(valid_df) > 0 and label_col in df.columns:
        y_true = valid_df[label_col]
        y_pred = valid_df["mistral_nemo_pred"].astype(int)

        kappa = cohen_kappa_score(y_true, y_pred)
        print("\n" + "=" * 50)
        print(f"  RESULTS SUMMARY — {model_name}")
        print("=" * 50)
        print(f"  Total comments:      {len(df)}")
        print(f"  Successfully parsed: {len(valid_df)} ({len(valid_df)/len(df)*100:.1f}%)")
        print(f"  Cohen's Kappa (κ):   {kappa:.4f}")
        print("-" * 50)
        print(
            classification_report(
                y_true, y_pred, target_names=["Neutral (0)", "Biased (1)"]
            )
        )


# CLI Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-Shot Regional Bias Annotation using Mistral Nemo LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Annotate with Mistral-Nemo
  python src/zero_shot/inference_nemo.py \\
      --model mistralai/Mistral-Nemo-Instruct-2407 \\
      --data_path data/raw/test_comments.csv \\
      --output results/annotation_results_mistral_nemo.csv

  # Resume an interrupted execution run
  python src/zero_shot/inference_nemo.py \\
      --model mistralai/Mistral-Nemo-Instruct-2407 \\
      --data_path data/raw/test_comments.csv \\
      --output results/annotation_results_mistral_nemo.csv \\
      --resume
        """,
    )
    parser.add_argument("--model", type=str, default="mistralai/Mistral-Nemo-Instruct-2407", help="HuggingFace model ID")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--comment_col", type=str, default="comment", help="Comment column name")
    parser.add_argument("--label_col", type=str, default="level-1", help="Ground truth label column name")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--gpu_id", type=int, default=1, help="GPU device index")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output")
    args = parser.parse_args()

    # Set GPU and print info
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print_gpu_info()

    # Run annotation
    run_zero_shot(
        model_name=args.model,
        data_path=args.data_path,
        output_path=args.output,
        comment_col=args.comment_col,
        label_col=args.label_col,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        resume=args.resume,
    )
