import os
import subprocess
import sys

def run_command(command: list, description: str):
    """Helper function to execute shell commands cleanly with real-time logging."""
    print(f"\n========================================\n[RUNNING] {description}\n========================================")
    try:
        # Runs the command and streams the output directly to your terminal
        process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr, text=True)
        process.wait()
        if process.returncode != 0:
            print(f"\n[ERROR] {description} failed with exit code {process.returncode}")
            sys.exit(1)
    except Exception as e:
        print(f"\n[EXCEPTION] Failed to execute command: {e}")
        sys.exit(1)

def build_automated_llamacpp_pipeline():
    # ----------------------------------------------------
    # Configuration & Paths
    # ----------------------------------------------------
    LLAMA_BIN_DIR = "./llama.cpp/build/bin"
    BASE_MODEL_F16 = "./base_mistral_f16.gguf"
    ADAPTER_DIR = "./optimized_mistral_weights"
    ADAPTER_GGUF = os.path.join(ADAPTER_DIR, "mistral-persona-adapter.gguf")
    
    MERGED_F16 = "./base_merged_f16.gguf"
    FINAL_QUANT_Q4 = "./mistral_persona_perfect_q4.gguf"

    print("Initializing Automated Edge Deployment Pipeline...")

    # 1. Verification of Build Binaries
    if not os.path.exists(os.path.join(LLAMA_BIN_DIR, "llama-export-lora")) or \
       not os.path.exists(os.path.join(LLAMA_BIN_DIR, "llama-quantize")):
        print(f"[ERROR] Could not find llama.cpp binaries in {LLAMA_BIN_DIR}. Please compile llama.cpp first.")
        sys.exit(1)

    # 2. Check for Base Model (Download if missing)
    if not os.path.exists(BASE_MODEL_F16):
        print(f"[INFO] {BASE_MODEL_F16} not found locally.")
        
        # Pulling down the base unquantized f16 file using huggingface-cli
        download_cmd = [
            "huggingface-cli", "download", 
            "MaziyarPanahi/Mistral-7B-v0.3-GGUF", 
            "Mistral-7B-v0.3.F16.gguf", 
            "--local-dir", ".", 
            "--local-dir-use-symlinks", "False"
        ]
        run_command(download_cmd, "Downloading Base Mistral-7B FP16 GGUF from Hugging Face")
        
        # Rename downloaded file to match our pipeline naming standard
        if os.path.exists("./Mistral-7B-v0.3.F16.gguf"):
            os.rename("./Mistral-7B-v0.3.F16.gguf", BASE_MODEL_F16)
    else:
        print(f"[SUCCESS] Found existing base model: {BASE_MODEL_F16}")

    # 3. Verify Adapter Weights Exist
    if not os.path.exists(ADAPTER_GGUF):
        print(f"[ERROR] Missing adapter weights file at: {ADAPTER_GGUF}")
        sys.exit(1)

    # 4. Step 1: Execute High-Precision On-Disk Structural Merge
    if not os.path.exists(MERGED_F16):
        merge_cmd = [
            os.path.join(LLAMA_BIN_DIR, "llama-export-lora"),
            "-m", BASE_MODEL_F16,
            "--lora", ADAPTER_GGUF,
            "-o", MERGED_F16
        ]
        run_command(merge_cmd, "Executing full-precision matrix merge (base + adapter) on disk")
    else:
        print(f"[SKIP] Merged FP16 file already exists at: {MERGED_F16}")

    # 5. Step 2: Quantize Merged Model down to 4-bit (Q4_K_M)
    quant_cmd = [
        os.path.join(LLAMA_BIN_DIR, "llama-quantize"),
        MERGED_F16,
        FINAL_QUANT_Q4,
        "Q4_K_M"
    ]
    run_command(quant_cmd, "Compressing fully merged model down to 4-Bit (Q4_K_M)")

    # ----------------------------------------------------
    # Production Ready Summary
    # ----------------------------------------------------
    print("\n========================================")
    print("[SUCCESS] PIPELINE PROCESSING COMPLETE!")
    print(f"-> Uncompressed baseline: {BASE_MODEL_F16}")
    print(f"-> Full precision merge:  {MERGED_F16}")
    print(f"-> Production edge model:  {FINAL_QUANT_Q4}")
    print("========================================")
    print("\nYou can now run your accelerated GPU deployment using:")
    print(f"{os.path.join(LLAMA_BIN_DIR, 'llama-cli')} -m {FINAL_QUANT_Q4} -ngl 99 -r \"<|im_end|>\" --no-conversation -p \"...\"")

if __name__ == "__main__":
    build_automated_llamacpp_pipeline()