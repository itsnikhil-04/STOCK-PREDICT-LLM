import os
import subprocess

# Define paths
DATA_PATH = "/content/drive/MyDrive/Minor/stock_prices.csv"
OUTPUT_PATH = "tokenized_data"

# Menu options
def main():
    print("\nStock Prediction Project Runner 🚀")
    print("1. Prepare Data")
    print("2. Fine-tune Model")
    print("3. Run Predictions")
    print("4. Run All")
    choice = input("Select an option (1-4): ")

    if choice == "1":
        prepare_data()
    elif choice == "2":
        fine_tune_model()
    elif choice == "3":
        run_predictions()
    elif choice == "4":
        run_all()
    else:
        print("❌ Invalid option! Please choose between 1-4.")

def prepare_data():
    print("\n📌 Preparing data...")
    command = f"python prepare_data.py --data_path {DATA_PATH} --output_path {OUTPUT_PATH}"
    subprocess.run(command, shell=True)

def fine_tune_model():
    print("\n📌 Fine-tuning Llama 2 model...")
    command = "python fine_tune_llama2.py"
    subprocess.run(command, shell=True)

def run_predictions():
    print("\n📌 Running stock predictions...")
    command = "python predict.py"
    subprocess.run(command, shell=True)

def run_all():
    prepare_data()
    fine_tune_model()
    run_predictions()

if __name__ == "__main__":
    main()
