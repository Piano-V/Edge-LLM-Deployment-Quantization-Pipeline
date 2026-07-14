import os
import time
import torch
from transformers import TrainingArguments, Trainer, TrainerCallback
from datasets import load_dataset
from initialize_pipeline import setup_low_memory_pipeline
import wandb

# 1. Initialize System Telemetry Trackers
wandb.init(
    project="systems-edge-llm", 
    name="mistral-converged-guanaco-run",
    config={
        "architecture": "Mistral-7B-v0.3",
        "optimization": "QLoRA 4-bit + DeepSpeed Stage 2",
        "dataset": "openassistant-guanaco",
        "max_steps": 50
    }
)

MODEL_ID = "mistralai/Mistral-7B-v0.3"

# 2. Boot the unquantized 4-bit low-memory base structure
model, tokenizer = setup_low_memory_pipeline(MODEL_ID)

# 3. Ingest and Stream conversational data matrices
print("Streaming instruction data vectors...")
dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")

def tokenize_and_format_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

# Select a 500-sample matrix block to guarantee data diversity over 150 steps
tokenized_dataset = dataset.select(range(500)).map(
    tokenize_and_format_function,
    batched=True,
    remove_columns=dataset.column_names
)

# 4. Configure DeepSpeed Optimization Arguments
training_args = TrainingArguments(
    output_dir="./optimized_mistral_weights",
    per_device_train_batch_size=1,       
    gradient_accumulation_steps=4,      # Simulates stable batch size of 4 via accumulation
    learning_rate=2e-4,
    lr_scheduler_type="cosine",          # Cosine decay prevents step shock
    warmup_steps=10,                     # Gradual learning rate warmup
    logging_steps=5,                    # Clean WandB tracking logging cadence
    max_steps=50,                       # Updated to match the targeted 150-step horizon
    bf16=True,                           
    deepspeed="ds_config.json",          # Shards parameters to system RAM over PCIe
    report_to="wandb",                  
    logging_first_step=True,
    remove_unused_columns=False
)

# 5. Advanced Thermal Management Subsystem
class AggressiveThermalStabilizer(TrainerCallback):
    def __init__(self, mandatory_cooldown=1.5):
        self.cooldown = mandatory_cooldown

    def on_step_end(self, args, state, control, **kwargs):
        # Force the CPU to pause execution until the GPU finishes all active matrix math
        torch.cuda.synchronize()
        # Flush the VRAM memory allocation fragmentation maps
        torch.cuda.empty_cache()
        # Mandatory rest period to allow the laptop cooling solution to clear the heat spikes
        time.sleep(self.cooldown)

# Initialize our specialized hardware pacing guard
thermal_callback = AggressiveThermalStabilizer(mandatory_cooldown=1.5)

# 6. Initialize Server Trainer Instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    callbacks=[thermal_callback] 
)

# 7. Run Training Matrix Optimization
print("[SYSTEM] Launching 50-step DeepSpeed convergence sequence...")
trainer.train()
print("[SUCCESS] Fine-tuning pass complete. Check WandB for the descending loss curve.")