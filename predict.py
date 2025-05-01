import torch
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model
def load_model(model_path="./fine_tuned_llama2"):
    print("[📌] Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

# Generate structured BUY, SELL, HOLD predictions
def predict_stock_movement(prompt, model, tokenizer, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    raw_prediction = tokenizer.decode(outputs[0], skip_special_tokens=False).upper()
    
    print(f"[🧠] Raw Model Output: {raw_prediction}")  # Debugging output

    if "BUY" in raw_prediction:
        return "BUY"
    elif "SELL" in raw_prediction:
        return "SELL"
    return "HOLD"  # Default to HOLD

# Generate a summary from CSV row
def generate_stock_summary(row):
    return (f"Stock: {row['Stock']}. Open: {row['Open']}, High: {row['High']}, "
            f"Low: {row['Low']}, Close: {row['Close']}, Volume: {row['Volume']}. "
            f"What is the recommended action? (BUY, SELL, or HOLD)")

# Predict from CSV file
def predict_from_csv(csv_path):
    print(f"[📈] Loading stock data from {csv_path}...")

    df = pd.read_csv(csv_path)

    if "summary" not in df.columns:
        print("[⚠️] No 'summary' column found. Generating summaries from stock data.")
        df["summary"] = df.apply(generate_stock_summary, axis=1)

    model, tokenizer = load_model()

    print("[🚀] Generating predictions...")
    df["predicted_movement"] = df["summary"].apply(lambda x: predict_stock_movement(x, model, tokenizer))

    output_path = "predictions.csv"
    df.to_csv(output_path, index=False)
    print(f"[✅] Predictions saved to {output_path}")

# Predict from JSONL file
def predict_from_jsonl(jsonl_path):
    print(f"[📈] Loading stock data from {jsonl_path}...")

    model, tokenizer = load_model()

    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]

    print("[🚀] Generating predictions...")
    for entry in data:
        if "summary" in entry:
            entry["predicted_movement"] = predict_stock_movement(entry["summary"], model, tokenizer)
        else:
            print("[⚠️] No 'summary' field found in JSONL file.")

    output_path = "predictions.jsonl"
    with open(output_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    
    print(f"[✅] Predictions saved to {output_path}")

# Main function
def main():
    print("🚀 Stock Movement Prediction 🚀")
    choice = input("Enter 1 to predict manually or 2 to predict from a file: ")

    model, tokenizer = load_model()

    if choice == "1":
        prompt = input("Enter stock summary for prediction: ")
        prediction = predict_stock_movement(prompt, model, tokenizer)
        print("📊 Predicted Movement:", prediction)

    elif choice == "2":
        file_type = input("Enter file type (1 for CSV, 2 for JSONL): ")
        file_path = input("Enter the path to the file: ")

        if file_type == "1":
            predict_from_csv(file_path)
        elif file_type == "2":
            predict_from_jsonl(file_path)
        else:
            print("❌ Invalid file type. Please choose 1 (CSV) or 2 (JSONL).")

    else:
        print("❌ Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
