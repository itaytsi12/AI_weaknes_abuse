import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time 
import gc # For memory cleanup

# --- 1. Model Configuration: CodeLlama-7B-Instruct ---
model_id = "codellama/CodeLlama-7b-Instruct-hf" 

# Initialize variables
model = None
tokenizer = None

try:
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 2. Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 3. Load Model with 4-bit Quantization (Essential for VRAM saving)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16, 
        load_in_4bit=True
    )
    print("CodeLlama Model loaded successfully with GPU support (4-bit quantization).")

except Exception as e:
    # Fallback to CPU if GPU loading fails
    print(f"Error loading model with GPU configuration. Falling back to CPU. Error: {e}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
    
# --- 2. Prompt Definition (Instruction Format) ---
instruction = "Write a complete Python class called 'SecureFileUploader' that includes a function to upload a file to a remote server using the 'requests' library, ensuring the connection uses SSL verification and requires an API key in the header."

prompt = f"### Instruction:\n{instruction}\n\n### Response:"

# --- 3. Generation ---
start_time = time.time()
inputs = tokenizer(prompt, return_tensors="pt").to(model.device) 

outputs = model.generate(
    **inputs,
    max_new_tokens=500,
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
)

end_time = time.time()
full_output = tokenizer.decode(outputs[0], skip_special_tokens=True) 

# --- 4. Output and Extraction ---
print(f"Time taken for generation: {end_time - start_time:.2f} seconds")
print("\n--- Generated Output (Full) ---")
print(full_output)

# Extract generated code based on CodeLlama's template
try:
    generated_code = full_output.split("### Response:")[-1].strip()
    print("\n--- Extracted Code ---")
    print(generated_code)
except IndexError:
    print("\nCould not cleanly split the output, check the full output above.")


# ===============================================
# === 5. GPU MEMORY CLEANUP ===
# ===============================================
if model is not None and tokenizer is not None:
    print("\n--- Cleaning up GPU Memory ---")
    
    del model
    del tokenizer
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    
    gc.collect() 
    print("Cleanup complete. GPU VRAM should be freed.")