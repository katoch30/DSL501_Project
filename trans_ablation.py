#!/usr/bin/env python3

import os
import csv
import logging
import re
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

PROJECT_DIR = '/data3/ritika_project/ml'
BASE_MODEL_PATH = "/data3/ritika_project/ml/TowerBase-7B-v0.1"
ADAPTER_DIR = os.path.join(PROJECT_DIR, "towerbase-7b-emotion-novel_prompt-192")

DATA_DIR = os.path.join(PROJECT_DIR, "data")
DEV_EN_VAD = os.path.join(DATA_DIR, "dev_txt", "dev_with_ids_vad.en")
DEV_FR = os.path.join(DATA_DIR, "dev_txt", "dev_with_ids.fr")
TEST_EN_VAD = os.path.join(DATA_DIR, "test_txt", "test_with_ids_vad.en")
TEST_FR = os.path.join(DATA_DIR, "test_txt", "test_with_ids.fr")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "ablation_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_INPUT_TOKENS = 192
MAX_NEW_TOKENS = 96
BATCH_SIZE = 4
NUM_BEAMS = 5
REPETITION_PENALTY = 1.2
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_vad_text_data(en_path: str, fr_path: str) -> List[Dict]:
    data = []
    if not (os.path.exists(en_path) and os.path.exists(fr_path)):
        logging.error("Missing VAD files")
        return data

    with open(en_path, "r", encoding="utf-8") as fe, open(fr_path, "r", encoding="utf-8") as ff:
        en_lines = fe.readlines()
        fr_lines = ff.readlines()

    n = min(len(en_lines), len(fr_lines))
    for i in range(n):
        en = en_lines[i].strip()
        fr = fr_lines[i].strip()

        m_en = re.match(r"^([\w\-]+)\s+V:([\d\.-]+)\s+A:([\d\.-]+)\s+D:([\d\.-]+)\s+(.*)$", en)
        m_fr = re.match(r"^([\w\-]+)\s+(.*)$", fr)

        if m_en and m_fr and m_en.group(1) == m_fr.group(1):
            try:
                data.append({
                    "AudioID": m_en.group(1),
                    "Valence": float(m_en.group(2)),
                    "Arousal": float(m_en.group(3)),
                    "Dominance": float(m_en.group(4)),
                    "EnglishText": m_en.group(5),
                    "FrenchText": m_fr.group(2),
                })
            except:
                pass
    return data


def prompt_instruction_only(sample: Dict) -> str:
    return (
        f"Translate the following English text to French, preserving its emotion.\n"
        f"English: {sample['EnglishText']}\nFrench:"
    )


def prompt_numbers_only(sample: Dict) -> str:
    v = sample["Valence"]
    a = sample["Arousal"]
    d = sample["Dominance"]
    return f"[VAL:{v:.2f} ARO:{a:.2f} DOM:{d:.2f}] {sample['EnglishText']}"


def load_model_and_tokenizer(adapter_dir: str):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # load base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    base.config.pad_token_id = tokenizer.pad_token_id

    # load adapter if present
    if os.path.exists(adapter_dir):
        dir_files = set(os.listdir(adapter_dir))
        if "adapter_config.json" in dir_files or "adapter_model.safetensors" in dir_files:
            model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
        else:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    adapter_dir,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
            except:
                model = base
    else:
        model = base

    model.eval()
    return model, tokenizer


def extract_french_from_output(text: str) -> str:
    marker = "French:"
    if marker in text:
        out = text.split(marker, 1)[1]
    else:
        out = text

    stops = ["\nEnglish:", "\n\nEnglish:", "\nFrench:", "<eos>", "</s>"]
    idxs = [out.find(s) for s in stops if s in out]
    if idxs:
        out = out[:min(idxs)]
    return out.strip()


@torch.no_grad()
def generate_for_prompts(model, tokenizer, prompts: List[str]) -> List[str]:
    results = []
    device = torch.device(model.device) if hasattr(model, "device") else torch.device("cuda")

    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating"):
        batch = prompts[i:i + BATCH_SIZE]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_TOKENS)
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        dec = tokenizer.batch_decode(out, skip_special_tokens=False)
        for d in dec:
            results.append(extract_french_from_output(d))

    return results


def process_and_save(name: str, samples: List[Dict], prompt_fn, model, tokenizer, out_csv: str):
    prompts = [prompt_fn(s) for s in samples]
    trans = generate_for_prompts(model, tokenizer, prompts)

    fields = ["id", "AudioID", "EnglishText", "Valence", "Arousal", "Dominance", "TranslatedFrench", "GroundTruthFrench"]
    Path(os.path.dirname(out_csv)).mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, s in enumerate(samples):
            writer.writerow({
                "id": i,
                "AudioID": s.get("AudioID", ""),
                "EnglishText": s.get("EnglishText", ""),
                "Valence": s.get("Valence", ""),
                "Arousal": s.get("Arousal", ""),
                "Dominance": s.get("Dominance", ""),
                "TranslatedFrench": trans[i] if i < len(trans) else "",
                "GroundTruthFrench": s.get("FrenchText", ""),
            })


def main():
    model, tokenizer = load_model_and_tokenizer(ADAPTER_DIR)

    dev_samples = load_vad_text_data(DEV_EN_VAD, DEV_FR)
    test_samples = load_vad_text_data(TEST_EN_VAD, TEST_FR)

    if dev_samples:
        process_and_save("dev_instruction_only", dev_samples, prompt_instruction_only, model, tokenizer,
                         os.path.join(OUTPUT_DIR, "dev_instruction_only.csv"))
        process_and_save("dev_numbers_only", dev_samples, prompt_numbers_only, model, tokenizer,
                         os.path.join(OUTPUT_DIR, "dev_numbers_only.csv"))

    if test_samples:
        process_and_save("test_instruction_only", test_samples, prompt_instruction_only, model, tokenizer,
                         os.path.join(OUTPUT_DIR, "test_instruction_only.csv"))
        process_and_save("test_numbers_only", test_samples, prompt_numbers_only, model, tokenizer,
                         os.path.join(OUTPUT_DIR, "test_numbers_only.csv"))


if __name__ == "__main__":
    main()
