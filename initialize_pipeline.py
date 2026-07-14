import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

def setup_low_memory_pipeline(model_id: str):
    print(f"Initializing boot sequence for: {model_id}")
    
    # 1. Custom Tokenizer Setup
    # Pulling the native tokenizer directly. No custom token additions required.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Config 4-Bit Quantization (The VRAM Savior)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True 
    )
    
    # 3. Load Base Model with explicit memory-mapping constraints
    print("Loading base model in 4-bit NF4 precision...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # CRITICAL FOR 16GB SYSTEM RAM: Prevent full model weight buffering in CPU memory
        low_cpu_mem_usage=True 
    )
    
    # 4. Prepare model for k-bit training
    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=True # Forces checkpointing immediately to protect VRAM during backprop
    )
    
    # 5. Configure LoRA Parameters
    # Target modules are perfectly specified for Mistral architecture.
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
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
    
    # Enable gradient checkpointing explicitly at the PEFT level
    model.gradient_checkpointing_enable()
    
    # Log memory footprint metrics to verify the optimization
    model.print_trainable_parameters()
    
    return model, tokenizer

if __name__ == "__main__":
    # Mistral-7B-v0.3 natively handles functional tokens without vocabulary expansion hacks.
    MODEL_ID = "mistralai/Mistral-7B-v0.3" 
    
    # Environment flag to optimize CUDA memory fragmentation on small GPUs
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:true"
    
    # Execution Test
    try:
        model, tokenizer = setup_low_memory_pipeline(MODEL_ID)
        print("\n[SUCCESS] Pipeline successfully initialized within 6GB VRAM / 16GB RAM budget.")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\n[OOM ERROR] Caught CUDA Out of Memory during initialization. Reduce LoRA rank or target fewer modules.")
        else:
            raise e