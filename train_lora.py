#!/usr/bin/env python3
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='Salesforce/codegen-350M-mono', help='Base model to fine-tune')
    parser.add_argument('--train_data', default='data/example_train.jsonl', help='Path to JSONL train data')
    parser.add_argument('--output_dir', default='lora-checkpoint', help='Where to save LoRA checkpoint')
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    # Optional: Prepare for k-bit training (when using bitsandbytes) - skip for small models
    # prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    ds = load_dataset('json', data_files=args.train_data)

    def prepare(examples):
        prompts = [i + '\n' + o for i, o in zip(examples['input'], examples['output'])]
        tokens = tokenizer(prompts, truncation=True, padding='max_length', max_length=512)
        tokens['labels'] = tokens['input_ids'].copy()
        return tokens

    tokenized = ds.map(prepare, batched=True, remove_columns=ds['train'].column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_steps=200,
        fp16=False,  # set True if you have GPU
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized['train'], data_collator=data_collator)
    trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
