import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def merge_and_save_pipeline(base_model_path: str, adapter_path: str, export_path: str):
    print("Initializing high-efficiency CPU weight-merging sequence...")

    # 1. Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # 2. Load the base model EXPLICITLY on CPU to safeguard VRAM
    # We load in float16/bfloat16 precision to avoid precision loss before merging
    print("Loading base model into system RAM (bypassing GPU VRAM)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"}, # Hard constraint forcing host RAM allocation
        low_cpu_mem_usage=True
    )

    # 3. Inject the trained adapter configurations
    print(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map={"": "cpu"}
    )

    # 4. Execute the mathematical merge
    # This mathematically updates the base weight matrices: W_new = W_base + (A x B)
    print("Executing matrix recombination (Merge & Unload)...")
    merged_model = model.merge_and_unload()

    # 5. Export the single, unified model asset back to disk
    print(f"Exporting production-ready weights to: {export_path}")
    merged_model.save_pretrained(export_path)
    tokenizer.save_pretrained(export_path)
    print("[SUCCESS] Model matrices successfully merged and exported on host infrastructure.")



if __name__ == "__main__":
    BASE_MODEL = "mistralai/Mistral-7B-v0.3"
    # Force absolute paths to prevent Hugging Face from confusing local folders with remote URLs
    ADAPTER_DIR = os.path.abspath("./optimized_mistral_weights")
    OUTPUT_DIR = os.path.abspath("./production_mistral_merged")

    # Run the export matrix pipeline
    if os.path.exists(ADAPTER_DIR):
        merge_and_save_pipeline(BASE_MODEL, ADAPTER_DIR, OUTPUT_DIR)
    else:
        print(f"[ERROR] Could not locate adapter weights at {ADAPTER_DIR}. Run training script first.") 