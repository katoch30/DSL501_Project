import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import Dataset
import os
import evaluate
from tqdm import tqdm 
import json 

# --- 1. Configuration ---


model_name = "mistral-7b"
base_model_path = "/data3/ritika_project/mistral-7b"
adapter_path = "/data3/ritika/mistral_training_output/mistral-7b-english-to-french"

# Option 2: TowerBase (uncomment the three lines below to use it)
# model_name = "towerbase-7b"
# base_model_path = "/data3/ritika_project/TowerBase-7B-v0.1"
# adapter_path = "/data3/ritika_project/towerbase_training_output/towerbase-7b-english-to-french"


# --- CHOOSE WHICH DATASET TO EVALUATE ON ---
# Option 1: The 'dev' set
dataset_name = "dev_set"
eval_en_path = "/data3/ritika/data/raw/dev_txt/dev/dev.en"
eval_fr_path = "/data3/ritika/data/raw/dev_txt/dev/dev.fr"

# Option 2: The 'test' set (uncomment the three lines below to use it)
# dataset_name = "test_set"
# eval_en_path = "/data3/ritika/data/raw/test/test.en"
# eval_fr_path = "/data3/ritika/data/raw/test/test.fr"


batch_size = 8


# --- 2. Load the Fine-Tuned Model ---

# BEST PRACTICE: Use BitsAndBytesConfig for 4-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print(f"Loading base model from: {base_model_path}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config, # Use the new config object
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # Use left padding for batch generation

print(f"Loading adapter from: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload() # Merge adapter for faster inference

print("Model ready for evaluation.")


# --- 3. Load the Evaluation Dataset ---

def load_evaluation_data(en_path, fr_path):
    """Loads parallel data and returns three lists: sources, references, and prompts."""
    if not os.path.exists(en_path) or not os.path.exists(fr_path):
        raise FileNotFoundError("Evaluation data files not found.")

    with open(en_path, 'r', encoding='utf-8') as f:
        sources = [line.strip() for line in f if line.strip()]
    with open(fr_path, 'r', encoding='utf-8') as f:
        references = [line.strip() for line in f if line.strip()]

    prompts = []
    for en_text in sources:
        prompt = f"### Instruction:\nTranslate the following English text to French: '{en_text}'\n\n### Response:\n"
        prompts.append(prompt)
            
    return sources, references, prompts

sources, references, prompts = load_evaluation_data(eval_en_path, eval_fr_path)
print(f"Loaded {len(sources)} sentences for evaluation.")


# --- 4. Generate Translations (Hypotheses) using BATCH PROCESSING ---

hypotheses = []
print("Generating translations for the evaluation set using batch processing...")

# Process the data in batches for huge speed improvements
for i in tqdm(range(0, len(prompts), batch_size)):
    # Get a batch of prompts
    batch_prompts = prompts[i:i+batch_size]
    
    # Tokenize the batch
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
    
    # Generate translations for the whole batch in parallel
    outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the batch of results
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Extract only the translated part for each item in the batch
    for response_text in decoded_outputs:
        try:
            response_start = response_text.find("### Response:") + len("### Response:")
            translation = response_text[response_start:].strip()
            hypotheses.append(translation)
        except:
            hypotheses.append("")

print("Translation generation complete.")


# --- 5. Calculate Metrics ---

print("Calculating BLEU score...")
bleu = evaluate.load("sacrebleu")
bleu_results = bleu.compute(predictions=hypotheses, references=references)
print(f"BLEU Score: {bleu_results['score']:.2f}")

print("\nCalculating COMET score...")
# COMET requires the original English sentences as well
comet = evaluate.load("comet")
comet_results = comet.compute(predictions=hypotheses, references=references, sources=sources)
print(f"COMET Score: {comet_results['mean_score']:.4f}")


# --- 6. Save Results to File ---

# Create a dictionary to hold the results
results_data = {
    "model_name": model_name,
    "dataset_name": dataset_name,
    "bleu_score": round(bleu_results['score'], 2),
    "comet_score": round(comet_results['mean_score'], 4),
    "translations": []
}

# Add the source, reference, and hypothesis for each sentence
for i in range(len(sources)):
    results_data["translations"].append({
        "source_english": sources[i],
        "reference_french": references[i],
        "model_translation": hypotheses[i]
    })

# Define the output filename
output_filename = "init_scores.json"

print(f"\nSaving results to {output_filename}...")
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(results_data, f, ensure_ascii=False, indent=4)

print("--- Evaluation Complete ---")

