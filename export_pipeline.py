import os
import subprocess
import sys

def run_command(command: list, description: str):
    """Execute command and stream output."""
    print(f"\n[RUNNING] {description}\n")
    try:
        process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr, text=True)
        process.wait()
        if process.returncode != 0:
            print(f"\n[ERROR] {description} failed with exit code {process.returncode}")
            sys.exit(1)
    except Exception as e:
        print(f"\n[EXCEPTION] Failed to execute command: {e}")
        sys.exit(1)

def build_automated_llamacpp_pipeline():
    # Paths configuration
    LLAMA_BIN_DIR = "./llama.cpp/build/bin"
    BASE_MODEL_F16 = "./base_mistral_f16.gguf"
    ADAPTER_DIR = "./optimized_mistral_weights"
    ADAPTER_GGUF = os.path.join(ADAPTER_DIR, "mistral-persona-adapter.gguf")
    
    MERGED_F16 = "./base_merged_f16.gguf"
    FINAL_QUANT_Q4 = "./mistral_persona_perfect_q4.gguf"

    print("Starting quantization pipeline...")

    # Verify compiled binaries exist
    if not os.path.exists(os.path.join(LLAMA_BIN_DIR, "llama-export-lora")) or \
       not os.path.exists(os.path.join(LLAMA_BIN_DIR, "llama-quantize")):
        print(f"[ERROR] Could not find llama.cpp binaries in {LLAMA_BIN_DIR}. Compile llama.cpp first.")
        sys.exit(1)

    # Download base model if not cached
    if not os.path.exists(BASE_MODEL_F16):
        print(f"{BASE_MODEL_F16} not found locally.")
        
        # Download using huggingface-cli
        download_cmd = [
            "huggingface-cli", "download", 
            "MaziyarPanahi/Mistral-7B-v0.3-GGUF", 
            "Mistral-7B-v0.3.F16.gguf", 
            "--local-dir", ".", 
            "--local-dir-use-symlinks", "False"
        ]
        run_command(download_cmd, "Downloading Base Mistral-7B FP16 GGUF")
        
        # Standardize filename
        if os.path.exists("./Mistral-7B-v0.3.F16.gguf"):
            os.rename("./Mistral-7B-v0.3.F16.gguf", BASE_MODEL_F16)
    else:
        print(f"Found base model: {BASE_MODEL_F16}")

    # Ensure trained adapter is present
    if not os.path.exists(ADAPTER_GGUF):
        print(f"[ERROR] Missing adapter weights at: {ADAPTER_GGUF}")
        sys.exit(1)

    # Merge LoRA weights into base model
    if not os.path.exists(MERGED_F16):
        merge_cmd = [
            os.path.join(LLAMA_BIN_DIR, "llama-export-lora"),
            "-m", BASE_MODEL_F16,
            "--lora", ADAPTER_GGUF,
            "-o", MERGED_F16
        ]
        run_command(merge_cmd, "Merging base model and LoRA adapter")
    else:
        print(f"Merged FP16 file already exists at: {MERGED_F16}")

    # Quantize the merged model to 4-bit (Q4_K_M)
    quant_cmd = [
        os.path.join(LLAMA_BIN_DIR, "llama-quantize"),
        MERGED_F16,
        FINAL_QUANT_Q4,
        "Q4_K_M"
    ]
    run_command(quant_cmd, "Quantizing merged model to Q4_K_M")

    # Print summary
    print("\nProcessing complete:")
    print(f"- Base model: {BASE_MODEL_F16}")
    print(f"- Merged model: {MERGED_F16}")
    print(f"- Quantized model: {FINAL_QUANT_Q4}")
    print("\nRun deployment using:")
    print(f"{os.path.join(LLAMA_BIN_DIR, 'llama-cli')} -m {FINAL_QUANT_Q4} -ngl 99 --temp 0.7 -p \"...\"")

if __name__ == "__main__":
    build_automated_llamacpp_pipeline()