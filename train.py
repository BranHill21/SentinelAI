import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Hyperparameters
    # --------------------------------------------------------------------------
    # Model and Data
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 
    dataset_name = "your_org/aerospace_synthetic_instruct" # Replace with actual Hugging Face dataset or local path
    output_dir = "./sentinel-llm-lora"
    
    # LoRA Hyperparameters (from PRD)
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # Training Hyperparameters
    per_device_train_batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    max_seq_length = 2048

    # --------------------------------------------------------------------------
    # 2. Tokenizer Setup
    # --------------------------------------------------------------------------
    print(f"Loading tokenizer: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right" # Important fix for fp16/bf16 training mapping

    # --------------------------------------------------------------------------
    # 3. Model & Quantization (QLoRA) Setup
    # --------------------------------------------------------------------------
    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading Base Model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    
    # --------------------------------------------------------------------------
    # 4. Prepare PEFT/LoRA Model
    # --------------------------------------------------------------------------
    print("Preparing model for parameter-efficient k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --------------------------------------------------------------------------
    # 5. Dataset Loading
    # --------------------------------------------------------------------------
    print(f"Loading dataset: {dataset_name}...")
    try:
        # Load real dataset
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"\n[Warning] Dataset '{dataset_name}' not found. Using a dummy self-instruct snippet to test the pipeline:\n{e}\n")
        from datasets import Dataset
        # Example Llama-3 instruction format snippet
        dummy_data = {
            "text": [
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is the tensile strength of Titanium Ti-6Al-4V according to MIL-STD-1500?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAccording to MIL-STD-1500, the minimum tensile strength of Ti-6Al-4V is typically 895 MPa (130,000 psi).<|eot_id|>"
            ] * 10 
        }
        dataset = Dataset.from_dict(dummy_data)
    
    # --------------------------------------------------------------------------
    # 6. Training Arguments & Trainer Initialization
    # --------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=1, # Adjust based on dataset size for real training
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        bf16=True, # Optimal for Ampere/Hopper (A100/H100)
        optim="paged_adamw_8bit", # Saves memory states to avoid OOM
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="tensorboard",
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # --------------------------------------------------------------------------
    # 7. Execute Training loop
    # --------------------------------------------------------------------------
    print("🚀 Starting Training Loop...")
    trainer.train()

    # --------------------------------------------------------------------------
    # 8. Save Final Model Adapter
    # --------------------------------------------------------------------------
    print(f"✅ Training complete. Saving finalized LoRA adapter to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
