import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import cohen_kappa_score, classification_report

# Direct local directory module layout configuration lookups
from config import MODEL_LIST, MAX_SEQ_LENGTH, MAX_NEW_TOKENS
from utils import (
    setup_env,
    build_chat_prompts,
    build_gemma_prompts,
    extract_label,
    extract_response_text,
    repair_csv,
    print_gpu_info,
)


# Primary Dynamic Zero-Shot Framework Runner.
def run_zero_shot_pipeline(model_alias, data_path, output_path, batch_size, backup_every, resume):
    model_hf_id = MODEL_ZOO[model_alias]
    print(f"\nInitialising Zero-Shot Pipeline for Model Zoo Target: {model_hf_id}")

    # Configure custom column markers dynamically based on the targeted model key
    raw_col = f"{model_alias.replace('-', '_')}_raw_output"
    pred_col = f"{model_alias.replace('-', '_')}_pred"

    # Setup target directories and backup paths
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    backup_file = out_file.parent / f"{out_file.stem}_backup.csv"

    # PROGRESS RESUMPTION STATE VERIFICATION
    if resume and out_file.exists():
        print(f"Found existing tracking state file: {out_file}. Verifying progress...")
        try:
            df = pd.read_csv(out_file)
        except pd.errors.ParserError:
            repair_csv(out_file)
            df = pd.read_csv(out_file)

        if pred_col in df.columns:
            processed_count = df[pred_col].notna().sum()
            start_index = (processed_count // batch_size) * batch_size
            print(f"Resuming pipeline execution boundary at entry index: {start_index}")
        else:
            start_index = 0
    else:
        print(f"Initiating clean zero-shot run from dataset: {data_path}")
        df = pd.read_csv(data_path)
        df[raw_col] = None
        df[pred_col] = None
        start_index = 0

    # Enforce strict integer formatting values on target ground truth labels if present
    if "level-1" in df.columns:
        df["level-1"] = pd.to_numeric(df["level-1"], errors="coerce").fillna(0).astype(int)

    comments_list = df["comment"].astype(str).tolist()

    if start_index >= len(df):
        print("Dataset is already fully annotated.")
    else:
        # MODEL INITIALIZATION ENVIRONMENT
        print(f"Loading Base Tokenizer Framework...")
        tokenizer = AutoTokenizer.from_pretrained(model_hf_id, padding_side="left", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Loading Generative Base Matrix Weights in native BFloat16 Precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_hf_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()

        # RUNTIME BATCH INFERENCE LOOPS
        print(f"Annotating {len(comments_list)} target frames (Batch Size: {batch_size})...")
        
        for i in tqdm(range(start_index, len(comments_list), batch_size)):
            batch_end = min(i + batch_size, len(comments_list))
            batch_comments = comments_list[i:batch_end]

            # Route prompt generation through architecture-specific format rules
            if "gemma" in model_alias:
                chat_prompts = build_gemma_prompts(batch_comments)
            else:
                chat_prompts = build_chat_prompts(batch_comments)

            texts = [
                tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in chat_prompts
            ]

            inputs = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH
            ).to(model.device)
            inputs.pop("token_type_ids", None)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.3,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode newly generated response text matrices
            generated_texts = tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            batch_raw = []
            batch_preds = []

            for text_response in generated_texts:
                response_text = extract_response_text(text_response)
                pred_label = extract_label(response_text)

                batch_raw.append(response_text)
                batch_preds.append(pred_label)

            df.loc[i : batch_end - 1, raw_col] = batch_raw
            df.loc[i : batch_end - 1, pred_col] = batch_preds

            # Execute periodic safety state progress saves to local storage disk
            batch_num = (i - start_index) // batch_size
            if batch_num % backup_every == 0 and batch_num > 0:
                df.to_csv(out_file, index=False)
                df.to_csv(backup_file, index=False)

        # Save finalized run tracking arrays
        df.to_csv(out_file, index=False)
        print(f"\nFinalised annotations successfully written to target destination: {out_file}")

    # EVALUATION ENGINE SUMMARY METRICS
    valid_df = df[df[pred_col].notna()]
    if len(valid_df) > 0 and "level-1" in df.columns:
        y_true = valid_df["level-1"].astype(int)
        y_pred = valid_df[pred_col].astype(int)

        kappa = cohen_kappa_score(y_true, y_pred)
        metrics_summary = classification_report(
            y_true, y_pred, target_names=["Neutral (0)", "Biased (1)"], zero_division=0, digits=4
        )

        print("\n" + "=" * 50)
        print(f"  ZERO-SHOT RUN PERFORMANCE SUMMARY — {model_alias.upper()}")
        print("=" * 50)
        print(f"  Target Foundation Checkpoint ID : {model_hf_id}")
        print(f"  Successfully Evaluated Entities : {len(valid_df)} / {len(df)}")
        print(metrics_summary)
        print(f"  Cohen's Kappa Operational (κ)   : {kappa:.4f}")
        print("-" * 50)

        # Output static summary validation log metrics text file
        txt_output_path = out_file.parent / f"{out_file.stem}_metrics_summary.txt"
        with open(txt_output_path, "w", encoding="utf-8") as f:
            f.write(f"ZERO-SHOT STUDY EVALUATION LOG FOR CHECKPOINT ID: {model_hf_id}\n")
            f.write("=" * 60 + "\n")
            f.write(metrics_summary)
            f.write(f"\nCohen's Kappa Performance Score (κ): {kappa:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Multi-Model Zero-Shot Inference Benchmark Study Pipeline Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_ZOO.keys()),
        help="Target model shorthand key string mapping properties within the configurations dictionary module",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path configuration values mapping location of source evaluation CSV target data",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Target output path location setup mapping where parsed score structures write out to local disk",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Inference matrix batch sizing distribution element parameters running on hardware",
    )
    parser.add_argument(
        "--backup_every",
        type=int,
        default=10,
        help="Checkpoint save frequencies parameters evaluating loops blocks boundary adjustments",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Toggle resumption flag checking tracking progress states to skip pre-annotated entries rows",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="Device indexing integer matching allocated GPU hardware targets",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    setup_env(str(args.gpu_id))
    print_gpu_info(0)

    run_zero_shot_pipeline(
        model_alias=args.model,
        data_path=args.data_path,
        output_path=args.output,
        batch_size=args.batch_size,
        backup_every=args.backup_every,
        resume=args.resume,
    )
