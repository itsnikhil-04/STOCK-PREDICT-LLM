import argparse
import os
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

def load_and_process_data(data_path, model_name="meta-llama/Llama-2-7b-hf"):
    # ✅ Ensure file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Error: The file {data_path} was not found.")
    
    # ✅ Load JSONL file
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"❌ JSON Decode Error: {e}")

    # ✅ Convert to Hugging Face Dataset
    dataset = Dataset.from_list(data)

    # ✅ Check if dataset has expected columns
    column_names = dataset.column_names
    if "prompt" not in column_names or "completion" not in column_names:
        raise ValueError(f"❌ Error: Expected 'prompt' and 'completion' columns, but found {column_names}")

    # ✅ Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

    # ✅ Tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["prompt"], padding="max_length", truncation=True, max_length=512)

    # ✅ Tokenize dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["prompt", "completion"])

    # ✅ Split into train and test sets
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    dataset = DatasetDict({
        "train": train_test_split["train"],
        "test": train_test_split["test"]
    })

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL dataset (e.g., stock_training_data.jsonl)")
    parser.add_argument("--output_path", type=str, default="tokenized_data", help="Path to save processed dataset")
    args = parser.parse_args()

    try:
        # ✅ Load and process data
        dataset = load_and_process_data(args.data_path)

        # ✅ Save dataset
        dataset.save_to_disk(args.output_path)
        print(f"✅ Tokenized dataset saved to {args.output_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
