from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# --- Configuration ---

model_id = "Unbabel/TowerBase-7B-v0.1"


save_directory = "/data3/ritika_project/TowerBase-7B-v0.1"


if __name__ == "__main__":
    
    if not os.path.exists(save_directory):
        print(f"Creating directory: {save_directory}")
        os.makedirs(save_directory)

    print(f"Downloading model '{model_id}' to '{save_directory}'...")

    # Download and save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_directory)

    # Download and save the model
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.save_pretrained(save_directory)

    print("\nDownload complete!")
    print(f"Model and tokenizer saved in: {save_directory}")
