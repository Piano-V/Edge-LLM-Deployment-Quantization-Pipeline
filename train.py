import os
import time
import torch
import wandb
from transformers import TrainingArguments, Trainer, TrainerCallback, DataCollatorForSeq2Seq
from datasets import load_dataset
from initialize_pipeline import setup_low_memory_pipeline

# Initialize WandB tracking
wandb.init(
    project="systems-edge-llm", 
    name="mistral-persona-stylization-alignment",
    config={
        "architecture": "Mistral-7B-v0.3",
        "optimization": "QLoRA 4-bit + DeepSpeed Stage 2",
        "dataset": "hieunguyenminh/roleplay",
        "max_steps": 15
    }
)

MODEL_ID = "mistralai/Mistral-7B-v0.3"

# Set up model and tokenizer with QLoRA & gradient checkpointing
model, tokenizer = setup_low_memory_pipeline(MODEL_ID)

# Apply Mistral native chat template
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)

# Load and format roleplay dataset
print("Loading dataset...")
dataset = load_dataset("hieunguyenminh/roleplay", split="train")

def tokenize_and_format_function(examples):
    formatted_texts = []
    
    for name, desc, raw_text in zip(examples["name"], examples["description"], examples["text"]):
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        
        # Parse dialogue pairs
        for i in range(0, len(lines) - 1, 2):
            user_msg = f"Adopt the persona of {name} ({desc}). How would you address a complex problem?"
            assistant_msg = lines[i+1]
            
            hf_messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ]
            
            templated_string = tokenizer.apply_chat_template(hf_messages, tokenize=False)
            formatted_texts.append(templated_string)

    outputs = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=256  # Keep sequences short to save VRAM
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_dataset = dataset.map(
    tokenize_and_format_function,
    batched=True,
    remove_columns=dataset.column_names
)

# Setup training arguments with DeepSpeed config
training_args = TrainingArguments(
    output_dir="./optimized_mistral_weights",
    per_device_train_batch_size=1,       
    gradient_accumulation_steps=4,      
    learning_rate=2e-5,                 
    lr_scheduler_type="cosine",          
    warmup_steps=3,                      
    logging_steps=1,                    
    max_steps=15,                       
    bf16=True,                           
    deepspeed="ds_config.json",          
    report_to="wandb",                  
    logging_first_step=True,
    remove_unused_columns=False,
    dataloader_pin_memory=False          
)

# Custom callback to prevent GPU overheating (for limited cooling environments)
class ThermalStabilizer(TrainerCallback):
    def __init__(self, mandatory_cooldown=2.0):
        self.cooldown = mandatory_cooldown

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        time.sleep(self.cooldown)

thermal_callback = ThermalStabilizer(mandatory_cooldown=2.0)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True
    ),
    callbacks=[thermal_callback] 
)

# Run training loop
print("Starting training loop...")
trainer.train()

print("Saving fine-tuned adapter weights...")
trainer.save_model("./optimized_mistral_weights") 

print("Fine-tuning complete.")