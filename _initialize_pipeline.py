import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

def setup_low_memory_pipeline(model_id: str):
    print(f"Initializing boot sequence for: {model_id}")
    
    # 1. Custom Tokenizer Setup
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Config 4-Bit Quantization (The VRAM Savior)
    # NF4 (NormalFloat4) is mathematically optimized for normally distributed neural network weights.
    # bfloat16 is used for compute to prevent underflow during backprop.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True # Quantizes the quantization constants themselves, saving ~0.4 GB VRAM
    )
    
    # 3. Load Base Model with explicit device mapping
    # device_map="auto" sharded load ensures we do not hit a host memory crash
    print("Loading base model in 4-bit NF4 precision...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # 4. Prepare model for k-bit training
    # This enables gradient checkpointing and freezes base model parameters
    model = prepare_model_for_kbit_training(model)
    
    # 5. Configure LoRA Parameters
    # We target all linear projections to maximize adapter capacity while keeping base weights frozen
    peft_config = LoraConfig(
        r=16,                       # Rank: Dimensionality of the low-rank matrices
        lora_alpha=32,              # Scaling factor for the adapter weights
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print("Injecting low-rank adapter (LoRA) matrices...")
    model = get_peft_model(model, peft_config)
    
    # Log memory footprint metrics to verify the optimization
    model.print_trainable_parameters()
    
    return model, tokenizer

if __name__ == "__main__":
    MODEL_ID = "mistralai/Mistral-7B-v0.3"
    
    # Execution Test
    try:
        model, tokenizer = setup_low_memory_pipeline(MODEL_ID)
        print("\n[SUCCESS] Pipeline successfully initialized within 6GB VRAM budget.")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\n[OOM ERROR] Caught CUDA Out of Memory during initialization. We need to tweak allocation.")
        else:
            raise e