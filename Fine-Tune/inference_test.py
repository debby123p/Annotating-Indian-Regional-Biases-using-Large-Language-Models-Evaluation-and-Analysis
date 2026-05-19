import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import classification_report, cohen_kappa_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MODEL_ZOO, MAX_SEQ_LENGTH, MAX_NEW_TOKENS_INFERENCE
from src.utils import setup_env, format_example, extract_label, print_gpu_info


# Production Inference and Evaluation Pipeline.
#
# Args:
#    model_alias - Shorthand key matching the config.py MODEL_ZOO dictionary
#    test_path   - Path to the 500-comment evaluation sample target CSV file
#    output_dir  - Directory where classification results will be saved
def run_evaluation_inference(model_alias, test_path, output_dir):
    model_hf_id = MODEL_ZOO[model_alias]
    adapter_path = Path("weights") / f"{model_alias}_best_adapter"
    
    print(f"\nInitialising Static Inference Pipeline for Model: {model_hf_id}")
    print(f"Targeting Production Adapter Checkpoint: {adapter_path}")
    
    if not adapter_path.exists():
        print(f"CRITICAL ERROR: Fine-tuned adapter weights not found at {adapter_path}.")
        print("Please verify that your cross-validation script executed successfully first.")
        return

    # LOAD TARGET EVALUATION DATASET
    print(f"Loading test evaluation sample from: {test_path}")
    df = pd.read_csv(test_path)
    
    # Ensure standard data formatting alignment strings are enforced
    df["comment"] = df["comment"].astype(str).str.strip()
    df = df.dropna(subset=["comment"]).reset_index(drop=True)
    
    # LOAD CONFIGURATION AND MODEL LAYERS 
    print(f"Loading tokenizer framework configuration...")
    tokenizer = AutoTokenizer.from_pretrained(model_hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading base generative matrix layers in standard bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_hf_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    print("Merging fine-tuned adaptive LoRA weight projections...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    # BATCH INFERENCE LOOP Engine
    print(f"Executing batch inference array over {len(df)} target sequences (Batch Size: 16)...")
    predictions = []
    raw_responses = []

    # Safe inference loop structure running across test frames 
    for i in tqdm(range(0, len(df), 16), desc=f"Evaluating {model_alias}"):
        batch = df.iloc[i : i + 16]
        texts = []
        
        for _, row in batch.iterrows():
            example = format_example(row, is_test=True)
            texts.append(
                tokenizer.apply_chat_template(
                    example["messages"], tokenize=False, add_generation_prompt=True
                )
            )
            
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH
        ).to(model.device)
        
        # Guard against leftover token type configurations inside attention steps
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_INFERENCE,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            
        # Isolate newly generated token sequences beyond slice boundaries
        decoded_batch = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        
        for text_response in decoded_batch:
            raw_output = text_response.strip()
            parsed_label = extract_label(raw_output)
            
            raw_responses.append(raw_output)
            predictions.append(parsed_label)

    # EXPORT RESULTS AND METRICS
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df[f"{model_alias}_raw_output"] = raw_responses
    df[f"{model_alias}_pred"] = predictions
    
    results_csv_file = output_path / f"{model_alias}_sample_500_results.csv"
    df.to_csv(results_csv_file, index=False)
    print(f"\nPredictions exported successfully to: {results_csv_file}")

    # Generate metric evaluation readouts if ground truth columns exist
    if "level-1" in df.columns:
        df["level-1"] = pd.to_numeric(df["level-1"], errors="coerce").fillna(0).astype(int)
        y_true = df["level-1"].tolist()
        
        kappa = cohen_kappa_score(y_true, predictions)
        metrics_report = classification_report(
            y_true, predictions, target_names=["Neutral (0)", "Biased (1)"], zero_division=0
        )
        
        print("\n" + "=" * 50)
        print(f"  EVALUATION SUMMARY — {model_alias.upper()}")
        print("=" * 50)
        print(metrics_report)
        print(f"  Cohen's Kappa (κ): {kappa:.4f}")
        print("-" * 50)
        
        # Save verification log summary metrics file out to file tree
        metrics_txt_file = output_path / f"{model_alias}_sample_500_metrics.txt"
        with open(metrics_txt_file, "w", encoding="utf-8") as f:
            f.write(f"MODEL BENCHMARK RUN: {model_hf_id}\n")
            f.write("=" * 50 + "\n")
            f.write(metrics_report)
            f.write(f"\nCohen's Kappa (κ): {kappa:.4f}\n")
        print(f"Performance report summary saved to: {metrics_txt_file}")


# CLI Arguments Entry Point Block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Downstream Sample Evaluation Inference Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_ZOO.keys()),
        help="Target model alias lookup shorthand key configured inside src/config.py",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="Path mapping to your custom target 500-comment evaluation CSV dataset file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation_runs",
        help="Directory output target path configuration mapping where raw scores write out",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=1,
        help="Target position coordinate element processing execution hardware pipelines",
    )
    args = parser.parse_args()

    # Hardware initialization configuration rules
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    setup_env(str(args.gpu_id))
    print_gpu_info(0)

    # Core engine downstream test execution call
    run_evaluation_inference(
        model_alias=args.model,
        test_path=args.test_path,
        output_dir=args.output_dir,
    )
