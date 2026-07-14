import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

def setup_low_memory_pipeline(model_id: str):
    print(f"Initializing boot sequence for: {model_id}")
    
    # Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization config (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True 
    )
    
    # Load base model
    print("Loading base model in 4-bit NF4 precision...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True # Prevents full model weight buffering in system RAM
    )
    
    # Prepare model for PEFT training
    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=True # Enable gradient checkpointing to reduce VRAM usage
    )
    
    # Configure LoRA
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
    
    print("Injecting LoRA adapters...")
    model = get_peft_model(model, peft_config)
    
    model.gradient_checkpointing_enable()
    
    # Print trainable parameters count
    model.print_trainable_parameters()
    
    return model, tokenizer

if __name__ == "__main__":
    MODEL_ID = "mistralai/Mistral-7B-v0.3" 
    
    # Avoid memory fragmentation on smaller GPUs
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:true"
    
    try:
        model, tokenizer = setup_low_memory_pipeline(MODEL_ID)
        print("\nPipeline successfully initialized.")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("\nCaught CUDA Out of Memory during initialization.")
        else:
            raise e