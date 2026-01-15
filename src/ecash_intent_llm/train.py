
import os
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

def train_entrypoint(config_path: str = "config/train_config.yaml"):
    print(f"🚀 Starting EasyCash Fine-Tuning Pipeline | Config: {config_path}")
    
    # 1. Load Configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model"]["base_model"]
    output_dir = cfg["model"]["adapter_output_dir"]

    # 2. Load Tokenizer with proper padding
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Important for fp16 training

    # 3. Quantization Config for QLoRA (Memory Efficient Training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 4. Load Base Model
    print("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=False # Silence warnings for gradient checkpointing
    )
    
    # Enable Gradient Checkpointing for memory saving
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 5. Define LoRA Config (Targeting Attention Layers)
    peft_config = LoraConfig(
        r=cfg["training"]["lora_r"], # Rank
        lora_alpha=cfg["training"]["lora_alpha"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",  
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ] # Targeting all linear layers for maximum expressivity
    )

    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters to verify LoRA
    print_trainable_parameters(model)

    # 6. Load Dataset (Assuming JSONL format)
    # real path: "data/processed/financial_intents_train.jsonl"
    # For fallback, we create a dummy dataset in memory if file missing
    try:
        dataset = load_dataset("json", data_files="data/processed/train.jsonl", split="train")
    except:
        print("⚠️ Warning: Train data not found. Using dummy dataset for syntax validation.")
        from datasets import Dataset
        dataset = Dataset.from_list([{"prompt": "bridge USDC", "completion": "..."}] * 10)

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=100, # Short run for demo
        learning_rate=float(cfg["training"]["learning_rate"]),
        fp16=True,
        logging_steps=1,
        optim="paged_adamw_8bit" # Page Optimizer to save GPU RAM
    )

    # 8. Initialize SFTTrainer (Supervised Fine-tuning)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        dataset_text_field="text", # Assuming dataset has 'text' column pre-formatted
        tokenizer=tokenizer,
        max_seq_length=1024
    )

    # 9. Train
    print("🔥 Commencing Training Run...")
    trainer.train()

    # 10. Save Artifacts
    print(f"Saving adapter to {output_dir}")
    trainer.save_model(output_dir)

def print_trainable_parameters(model):
    """
    Helper to calculate the number of trainable params
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

if __name__ == "__main__":
    train_entrypoint()
