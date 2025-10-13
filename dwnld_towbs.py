from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# --- Configuration ---
# The Hugging Face ID of the model you want to download
model_id = "Unbabel/TowerBase-7B-v0.1"

# The local directory where you want to save the model
# We will save it in the parent directory of your 'ml' folder for organization
save_directory = "/data3/ritika_project/TowerBase-7B-v0.1"

# --- Main Script ---
if __name__ == "__main__":
    # Ensure the save directory exists
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

    print("\n✅ Download complete!")
    print(f"Model and tokenizer saved in: {save_directory}")
