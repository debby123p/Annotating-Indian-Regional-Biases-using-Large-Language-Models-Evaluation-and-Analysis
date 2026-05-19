import os
import re

# Central mapping model aliases to HuggingFace IDs
MODEL_LIST = {
    "qwen-8b": "Qwen/Qwen3-8B",
    "qwen-32b": "Qwen/Qwen3-32B",
    "dseek-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "krutrim-2": "krutrim-ai-labs/Krutrim-2-instruct",
    "mixtral-moe": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mnemo-12b": "mistralai/Mistral-Nemo-Instruct-2407",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-27b": "google/gemma-3-27b-it",
}

# Unified Fine-Tuning & Inference Hyperparameters
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS_TRAIN = 150
MAX_NEW_TOKENS_INFERENCE = 256
SEED = 42
NUM_CHUNKS = 10
NUM_FOLDS = 5

# Standardized System Prompt
SYSTEM_PROMPT = (
    "You are an expert socio-linguist with extensive knowledge of the Indian subcontinent's "
    "diverse states, cultures, languages, and regional dynamics. Your task is to annotate comments "
    "based on the presence of Regional Bias.\n\n"
    "Definitions:\n"
    "- REGIONAL BIAS (1): Comments that reinforce or propagate biases about Indian states, regions, "
    "or their people based on linguistic, cultural, economic, political, or infrastructural aspects. "
    "The comments can reflect either positive or negative biases towards specific states or regions.\n"
    "- NON-REGIONAL BIAS (0): Comments that are neutral or factual without generalisations, or "
    "unrelated to regional characteristics."
)
