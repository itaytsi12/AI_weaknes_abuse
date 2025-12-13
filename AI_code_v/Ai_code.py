import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import time 
import gc # For memory cleanup
import argparse
try:
    import psutil
except Exception:
    psutil = None

# --- CLI & Model Configuration: parse early so CLI overrides work ---
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="codellama/CodeLlama-7b-Instruct-hf", help="Model id to use (overrides auto selection)")
parser.add_argument("--max_new_tokens", type=int, default=500, help="Max tokens to generate")
parser.add_argument("--do_sample", action="store_true", help="Enable sampling during generation")
parser.add_argument("--train", action="store_true", help="Run LoRA training script")
parser.add_argument("--train_data", default=None, help="Path to JSONL training file for LoRA (optional)")
args = parser.parse_args()

model_id = args.model

# Fallback model (smaller) when the environment doesn't have enough RAM
FALLBACK_MODEL = "distilgpt2"

# Initialize variables
model = None
tokenizer = None

try:
    # 1. Load Tokenizer (later reloaded if model_id changes to fallback)

    # 2. Check for GPU & available memory
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    available_gb = None
    if psutil is not None:
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        print(f"Available memory: {available_gb:.2f} GB")
    # If we're on CPU and memory is lower than ~12GB, use fallback
    if device == "cpu" and available_gb is not None and available_gb < 12:
        print("Warning: Low available RAM. Switching to a smaller fallback model to avoid OOM.")
        model_id = FALLBACK_MODEL
        # reload the tokenizer for the chosen fallback model
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 3. Load Model: use quantized GPU load only when CUDA is available
    if device == "cuda":
        # GPU path, load with 4-bit quantization to save VRAM
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype="float16"  # use dtype param (not torch_dtype)
        )
        print("CodeLlama Model loaded successfully with GPU support (4-bit quantization).")
    else:
        # CPU path: avoid 4-bit/GPU-specific args to prevent memory / mmap errors
        # Use low_cpu_mem_usage to reduce peak memory while loading
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        print("CodeLlama Model loaded for CPU. Note: this model may still require lots of RAM.")

except Exception as e:
    # Fallback to CPU if GPU loading fails
    print(f"Error loading model with GPU configuration. Falling back to CPU. Error: {e}")
    # if tokenizer not set for final model_id, try to load it
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        tokenizer = None
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
    
# --- CLI & Prompt Definition ---
# CLI args already parsed earlier
# ensure tokenizer is consistent with the final `model_id`
if tokenizer is None or tokenizer.name_or_path != model_id:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        pass

# --- 2. Prompt Definition (Instruction Format) ---
instruction = "Write a complete Python class called 'SecureFileUploader' that includes a function to upload a file to a remote server using the 'requests' library, ensuring the connection uses SSL verification and requires an API key in the header."

prompt = f"### Instruction:\n{instruction}\n\n### Response:"

# --- 3. Generation ---
start_time = time.time()
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

generate_kwargs = dict(
    **inputs,
    max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else 64,
    do_sample=bool(args.do_sample),
    temperature=0.6 if args.do_sample else 1.0,
    top_p=0.95 if args.do_sample else 1.0,
    eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None,
    pad_token_id=getattr(tokenizer, 'pad_token_id', None) or getattr(tokenizer, 'eos_token_id', None),
)

outputs = model.generate(**generate_kwargs)

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

# --- LoRA Fine-Tuning (opt-in) ---
if args.train:
    MODEL = "Salesforce/codegen-350M-mono"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)

    # Prepare model for k-bit training if using bnb or quantized
    # prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Dataset: load JSONL with fields "input" and "output"
    if args.train_data is None:
        raise ValueError("--train_data must be provided when --train is set")
    ds = load_dataset("json", data_files=args.train_data)

    def prepare(examples):
        prompts = ["<BOS>" + i + o for i, o in zip(examples["input"], examples["output"])]
        tokens = tokenizer(prompts, truncation=True, padding="max_length", max_length=512)
        return tokens

    tokenized = ds.map(prepare, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir="lora-checkpoint",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=200,
        fp16=True  # if you have GPU, otherwise omit
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized["train"], data_collator=data_collator)
    trainer.train()
    trainer.save_model("lora-trained")

    # demonstration inference after training (optional)
    base = MODEL
    tokenizer = AutoTokenizer.from_pretrained(base)
    base_model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
    model = PeftModel.from_pretrained(base_model, "lora-trained")  # path to saved PEFT folder

    text = "Your instruction or prompt here"
    tokens = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**tokens, max_new_tokens=128)
    print(tokenizer.decode(out[0], skip_special_tokens=True))