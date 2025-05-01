import os
import gc
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

# ✅ Suppress TensorFlow CUDA warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WANDB_DISABLED"] = "true"  # Disable W&B logs

# ✅ Load LLaMA-2 7B Model with 4-bit Quantization
MODEL_NAME = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # ✅ More stable than float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # ✅ Fix padding error

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# ✅ Enable Gradient Checkpointing & Training Mode
model.gradient_checkpointing_enable()
model.config.use_cache = False  # ✅ Fix 'use_cache=True' conflict
model.train()

# ✅ Apply LoRA (Memory-Efficient Fine-Tuning)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ✅ Load & Tokenize Training Data
DATASET_PATH = "stock_training_data.jsonl"
dataset = load_dataset("json", data_files=DATASET_PATH)
dataset = dataset["train"].train_test_split(test_size=0.2)

def tokenize_function(examples):
    """Tokenize input and set labels for loss calculation"""
    tokenized = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"][:]
    return tokenized

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["completion"])

# ✅ Ensure inputs require gradients
for param in model.parameters():
    if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
        param.requires_grad = True

# ✅ Reduce Batch Size & Enable Mixed Precision
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama2",
    per_device_train_batch_size=1,  # ✅ Reduce batch size to save memory
    gradient_accumulation_steps=16,  # ✅ Increase accumulation steps to reduce memory usage
    num_train_epochs=3,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    bf16=True,  # ✅ Use bfloat16 instead of fp16
    optim="paged_adamw_8bit",  # ✅ Efficient optimizer for quantized models
)

# ✅ Free GPU Memory Before Training
gc.collect()
torch.cuda.empty_cache()

# ✅ Start Fine-Tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"]
)

trainer.train()

# ✅ Save Fine-Tuned Model
model.save_pretrained("./fine_tuned_llama2")
tokenizer.save_pretrained("./fine_tuned_llama2")

print("✅ Fine-tuning complete! Model saved to './fine_tuned_llama2'")
