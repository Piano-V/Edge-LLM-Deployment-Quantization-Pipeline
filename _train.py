import os
import torch
from transformers import TrainingArguments, Trainer
from initialize_pipeline import setup_low_memory_pipeline
import wandb

# Initialize Weights & Biases Telemetry
wandb.init(project="systems-edge-llm", name="rtx-4050-deepspeed-optimization")

MODEL_ID = "mistralai/Mistral-7B-v0.3"

# 1. Initialize our custom optimized model & tokenizer
model, tokenizer = setup_low_memory_pipeline(MODEL_ID)

# 2. Mock Dataset generation for testing infrastructure throughput
# We generate random token vectors to simulate a heavy processing workflow
print("Generating streaming mock training telemetry...")
class MockDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, num_samples=20, seq_length=512):
        self.input_ids = torch.randint(0, tokenizer.vocab_size, (num_samples, seq_length))
        self.labels = self.input_ids.clone()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}

train_dataset = MockDataset(tokenizer)

# 3. Training Arguments linked directly to our DeepSpeed JSON configuration
training_args = TrainingArguments(
    output_dir="./optimized_mistral_weights",
    per_device_train_batch_size=1,       # Kept at 1 due to 6GB VRAM constraint
    gradient_accumulation_steps=4,      # Simulates batch size 4 via accumulation
    learning_rate=2e-4,
    logging_steps=1,
    max_steps=5,                         # Low step ceiling just to test structural viability
    bf16=True,                           # Safe compute datatype for Ada Lovelace GPUs
    deepspeed="ds_config.json",          # Injecting the DeepSpeed system profile
    report_to="wandb",                  # Stream real-time performance to WandB
    logging_first_step=True
)

# 4. Initialize Trainer Architecture
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 5. Launch Training Pipeline
print("Launching DeepSpeed Orchestration...")
trainer.train()
print("[SUCCESS] Infrastructure successfully executed a distributed training step without OOMing!")