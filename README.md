# AI Weakness Abusive Research

This project contains a script at `AI_code_v/Ai_code.py` for running local model inference and optional LoRA fine-tuning.

Recommended small models for low-RAM machines:
- `distilgpt2` — very small, fastest, best for quick tests (not instruction-tuned).
- `gpt2` — small and fast, general text.
- `Salesforce/codegen-350M-mono` — small code-focused model, best for code tasks and LoRA fine-tuning.
- `bigscience/bloomz-560m` — instruction-tuned, better for instruction-to-code behavior but larger.

Example commands (from project root):

Activate venv and install requirements:

```bash
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run inference using `distilgpt2` with 64 tokens:

```bash
python AI_code_v/Ai_code.py --model distilgpt2 --max_new_tokens 64 --do_sample
```

Fine-tune with LoRA (requires `--train_data` and a GPU):

```bash
python AI_code_v/Ai_code.py --train --train_data path/to/train.jsonl --model Salesforce/codegen-350M-mono
```

Notes:
- `distilgpt2` and `gpt2` are not instruction-tuned; to get better instruction-following use BloomZ or fine-tune CodeGen.
- Ensure your machine has enough RAM or add swap if using larger models.
