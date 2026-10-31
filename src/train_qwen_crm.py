"""
train_qwen_crm.py
-----------------
Unsloth QLoRA Fine-Tuning Script for Qwen 2.5 3B Instruct.
Optimized for GPUs with 6GB VRAM constraints.

Workflow:
  1. Load Qwen 2.5 3B in 4-bit quantization with 1024 context window.
  2. Apply LoRA adapters targeting all attention & MLP projection layers.
  3. Format synthetic CRM training data into Qwen ChatML with explicit EOS tokens.
  4. Train using memory-efficient SFTTrainer (batch size 1, grad accumulation 4, adamw_8bit).
  5. Export fine-tuned weights directly into 4-bit GGUF format (q4_k_m) for local inference.
"""

import os
from pathlib import Path
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, TrainingArguments

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_FILE = ROOT_DIR / "data" / "langgraph_agents_train.jsonl"

# Fallback to synthetic_data_pipeline/data if not in root data/
if not DATA_FILE.exists():
    fallback_file = ROOT_DIR / "synthetic_data_pipeline" / "data" / "langgraph_agents_train.jsonl"
    if fallback_file.exists():
        DATA_FILE = fallback_file

EXPORT_DIR = ROOT_DIR / "model_export" / "qwen_crm_agent"

# ---------------------------------------------------------------------------
# 1. Model Initialization (Memory Optimized for <= 6GB VRAM)
# ---------------------------------------------------------------------------
max_seq_length = 1024  # Strict context ceiling to prevent VRAM spikes
load_in_4bit = True    # 4-bit quantization reduces base model footprint to ~2GB

print("[1/5] Loading Qwen 2.5 3B Instruct with 4-bit quantization...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",
    max_seq_length=max_seq_length,
    dtype=None,  # Auto-detected (Float16 on supported GPUs)
    load_in_4bit=load_in_4bit,
)

# ---------------------------------------------------------------------------
# 2. LoRA Adapter Configuration
# ---------------------------------------------------------------------------
print("[2/5] Configuring LoRA adapters and gradient checkpointing...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # 0 is optimized by Unsloth
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth specialized offloading saves ~30% VRAM
    random_state=3407,
    max_seq_length=max_seq_length,
)

# ---------------------------------------------------------------------------
# 3. Dataset ChatML Formatting
# ---------------------------------------------------------------------------
print(f"[3/5] Loading and formatting dataset from {DATA_FILE}...")
EOS_TOKEN = tokenizer.eos_token


def formatting_prompts_func(examples):
    """
    Formats multi-turn CRM agent examples into native Qwen ChatML:
    <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|><eos>
    """
    systems = examples["system"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    for system, user_input, output in zip(systems, inputs, outputs):
        text = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>"
            + EOS_TOKEN
        )
        texts.append(text)

    return {"text": texts}


dataset = load_dataset("json", data_files=str(DATA_FILE), split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# ---------------------------------------------------------------------------
# 4. Trainer Configuration (6GB VRAM Safe)
# ---------------------------------------------------------------------------
print("[4/5] Initializing SFTTrainer with 6GB VRAM safety settings...")
training_args = TrainingArguments(
    per_device_train_batch_size=1,       # Keeps single-forward peak memory minimal
    gradient_accumulation_steps=4,       # Simulates effective batch size of 4
    warmup_steps=5,
    max_steps=300,                       # Fixed step budget for efficient fine-tuning
    learning_rate=2e-4,
    fp16=True,                           # Half-precision training
    logging_steps=10,
    optim="adamw_8bit",                  # 8-bit optimizer saves ~75% optimizer VRAM
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=training_args,
)

# ---------------------------------------------------------------------------
# 5. Execution and GGUF Export
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[5/5] Starting QLoRA training run...")
    trainer_stats = trainer.train()

    print("\n--- Training Finished ---")
    print(f"Exporting model to 4-bit GGUF at: {EXPORT_DIR}")
    
    EXPORT_DIR.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        "model_export/qwen_crm_agent",
        tokenizer,
        quantization_method="q4_k_m",
    )
    print("GGUF export complete! Ready for local Ollama / llama.cpp inference.")

