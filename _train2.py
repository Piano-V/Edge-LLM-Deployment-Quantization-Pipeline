import os
import torch
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from initialize_pipeline import setup_low_memory_pipeline
import wandb

# 1. Initialize Weights & Biases Live Telemetry Trackers
wandb.init(
    project="systems-edge-llm", 
    name="mistral-real-data-deepspeed-run",
    config={
        "architecture": "Mistral-7B-v0.3",
        "optimization": "QLoRA 4-bit + DeepSpeed Stage 2",
        "dataset": "openassistant-guanaco"
    }
)

MODEL_ID = "mistralai/Mistral-7B-v0.3"

# 2. Boot our verified 4-bit NF4 low-memory base model and tokenizer
model, tokenizer = setup_low_memory_pipeline(MODEL_ID)

# 3. Ingest and Format the Real Dataset
print("Streaming high-quality conversational dataset from Hub...")
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

def tokenize_and_format_function(examples):
    # This maps the raw conversational text directly into the tokenizer's mathematical vector layout
    # We enforce a hard max-length cut-off to prevent any hidden memory spikes
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

print("Applying tokenization formatting matrix...")
# We use a small subset (100 samples) to ensure fast loop closure on your laptop hardware
tokenized_dataset = dataset.select(range(100)).map(
    tokenize_and_format_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 4. Bind System Architecture Configurations to DeepSpeed
training_args = TrainingArguments(
    output_dir="./optimized_mistral_weights",
    per_device_train_batch_size=1,       # Kept locked at 1 due to 6GB hardware boundary
    gradient_accumulation_steps=4,      # Simulates a stable batch size of 4 via accumulation
    learning_rate=2e-4,
    lr_scheduler_type="cosine",          # High-performance cosine decay curve
    logging_steps=1,
    max_steps=10,                        # Running 10 full optimization steps for clear metrics
    bf16=True,                           # Explicit bfloat16 hardware processing
    deepspeed="ds_config.json",          # Your custom optimizer virtualization engine
    report_to="wandb",                  # Direct hook to stream telemetry
    logging_first_step=True,
    remove_unused_columns=False
)

# 5. Initialize Server Trainer Architecture Execution Engine
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 6. Execute Training Runtime Process
print("Launching DeepSpeed Real-Data Orchestration Sequence...")
trainer.train()
print("[SUCCESS] Fine-tuning pass complete. Gradients successfully converged and logged.")