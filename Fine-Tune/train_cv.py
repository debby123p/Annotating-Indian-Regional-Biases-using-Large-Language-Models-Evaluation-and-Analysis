import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

from src.config import (
    MODEL_ZOO,
    MAX_SEQ_LENGTH,
    MAX_NEW_TOKENS_TRAIN,
    NUM_CHUNKS,
    NUM_FOLDS,
    SEED,
)
from src.utils import setup_env, format_example, print_gpu_info


# Primary Cross-Validation and Weight Extraction Loop.
#
# Args:
#    model_alias - Shorthand key matching the config.py MODEL_ZOO dictionary
#    data_path   - Path to the 25k row master training dataset CSV file
#    output_root - Target directory where fold checkpoints will be saved
def run_cross_validation(model_alias, data_path, output_root):
    model_hf_id = MODEL_ZOO[model_alias]
    print(f"\nInitialising Cross-Validation Pipeline for Model: {model_hf_id}")
    
    # LOAD AND PREPROCESS DATASET 
    print(f"Loading master dataset from: {data_path}")
    df = pd.read_csv(data_path)
    df["level-1"] = df["level-1"].astype(int)
    df["comment"] = df["comment"].astype(str).str.strip()
    df = df.dropna(subset=["comment"]).reset_index(drop=True)
    
    # Maintain maximum 25k balance rows as configured across benchmark runs
    if len(df) > 25000:
        df = df.sample(n=25000, random_state=SEED).reset_index(drop=True)
        
    # Standard 10-chunk stratified setup array splitting mechanics
    skf = StratifiedKFold(n_splits=NUM_CHUNKS, shuffle=True, random_state=SEED)
    chunks = [idx for _, idx in skf.split(df.index, df["level-1"])]
    
    # Setup metric trackers to identify the winning adapter parameters
    best_fold_idx = -1
    lowest_eval_loss = float("inf")
    
    # FOLD LOOP RUNTIME ENGINE 
    for fold in range(NUM_FOLDS):
        fold_num = fold + 1
        print(f"\n=== STARTING EXPERIMENTAL FOLD {fold_num} / {NUM_FOLDS} ===")
        
        fold_dir = Path(output_root) / model_alias / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Programmatic chunk windowing distribution logic
        test_indices_list = [(fold + i) % NUM_CHUNKS for i in range(5)]
        val_indices_list = [(fold + 5) % NUM_CHUNKS]
        train_indices_list = [(fold + 6 + i) % NUM_CHUNKS for i in range(4)]
        
        train_df = df.iloc[np.concatenate([chunks[i] for i in train_indices_list])].copy()
        val_df = df.iloc[chunks[val_indices_list[0]]].copy()
        
        # Load specific tokenizer configuration elements
        print(f"Loading tokenizer framework for: {model_hf_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_hf_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # Load underlying foundation generative matrix architecture
        print(f"Loading foundation model layers...")
        model = AutoModelForCausalLM.from_pretrained(
            model_hf_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # Wrap datasets into HuggingFace dataset classes using standardized utility formatters
        train_ds = Dataset.from_pandas(train_df).map(
            lambda x: {"text": tokenizer.apply_chat_template(format_example(x)["messages"], tokenize=False)}
        )
        val_ds = Dataset.from_pandas(val_df).map(
            lambda x: {"text": tokenizer.apply_chat_template(format_example(x)["messages"], tokenize=False)}
        )
        
        # Configure LoRA hyperparameters targeting linear project layers
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        )
        
        # Standardize training arguments parameters configuration mapping
        training_args = SFTConfig(
            output_dir=str(fold_dir / "checkpoints"),
            num_train_epochs=10,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=20,
            optim="adamw_8bit",
            warmup_ratio=0.03,
            max_grad_norm=0.3,
            lr_scheduler_type="cosine",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            max_length=MAX_SEQ_LENGTH,
            dataset_text_field="text",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            gradient_checkpointing=True,
        )
        
        # Initialize trainer wrapper
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            peft_config=peft_config,
            processing_class=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)],
        )
        
        print(f"Executing training loop sequence for Fold {fold_num}...")
        train_result = trainer.train()
        
        # Isolate and save current best fold runtime adapter state
        adapter_output_path = fold_dir / "best_adapter"
        trainer.save_model(str(adapter_output_path))
        
        # Access evaluation log metrics to track optimal performance metrics
        eval_metrics = trainer.evaluate()
        current_eval_loss = eval_metrics.get("eval_loss", float("inf"))
        print(f"Fold {fold_num} complete. Evaluation Loss reached: {current_eval_loss:.4f}")
        
        # Capture the index position yielding lowest cross-entropy variance loss
        if current_eval_loss < lowest_eval_loss:
            lowest_eval_loss = current_eval_loss
            best_fold_idx = fold_num
            
        # Clean down active VRAM arrays prior to initiating next loop sequences
        del model, trainer, train_ds, val_ds
        torch.cuda.empty_cache()
        gc.collect()
        
    # ISOLATE PRODUCTION WEIGHTS 
    print("\n" + "=" * 50)
    print("CROSS-VALIDATION ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"Optimal configuration isolated at Fold Index: {best_fold_idx}")
    print(f"Lowest achieved Evaluation Loss metrics: {lowest_eval_loss:.4f}")
    
    # Export winning weights path location configuration values
    source_best_adapter = Path(output_root) / model_alias / f"fold_{best_fold_idx}" / "best_adapter"
    production_weights_path = Path("weights") / f"{model_alias}_best_adapter"
    production_weights_path.mkdir(parents=True, exist_ok=True)
    
    # Reload model profile wrapper to export optimal structural checkpoints natively
    print(f"Isolating winning configuration from {source_best_adapter}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_hf_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    best_peft_model = PeftModel.from_pretrained(base_model, str(source_best_adapter))
    
    print(f"Exporting optimal performance adapter checkpoints natively to: {production_weights_path}")
    best_peft_model.save_pretrained(str(production_weights_path))
    
    print("Production adapter weights isolated. Pipeline step complete.")


# CLI Arguments Entry Block Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Multi-Model Stratified Cross-Validation Architecture Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_ZOO.keys()),
        help="Target configuration model shorthand key matching src/config.py dictionary keys",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path targeting your master training dataset CSV file source location",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/cv_benchmarks",
        help="Directory root folder structure path configuration mapping where raw runs save",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=1,
        help="Device position matrix targeting active GPU hardware allocations",
    )
    args = parser.parse_args()

    # Hardware verification runtimes
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    setup_env(str(args.gpu_id))
    print_gpu_info(0)

    # Core engine validation run execution
    run_cross_validation(
        model_alias=args.model,
        data_path=args.data_path,
        output_root=args.output_dir,
    )
