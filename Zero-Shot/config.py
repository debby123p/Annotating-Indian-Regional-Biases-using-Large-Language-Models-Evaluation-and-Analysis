import os

# Central Zero-Shot Model mapping aliases to HuggingFace IDs
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

# Standardized Global Zero-Shot Hyperparameters
MAX_SEQ_LENGTH = 2048
MAX_NEW_TOKENS = 512
SEED = 42

# Standardized Zero-Shot System Prompt
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
