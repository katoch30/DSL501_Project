import os
import csv
import re
import torch
import logging
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_DIR = '/data3/ritika_project/ml'
BASE_MODEL_PATH =  "/data3/ritika_project/ml/TowerBase-7B-v0.1"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "towerbase_output")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

EXPERIMENTS = ["novel_prompt"]

MAX_INPUT_TOKENS = 192
MAX_NEW_TOKENS = 96
BATCH_SIZE = 4
NUM_BEAMS = 5
REPETITION_PENALTY = 1.2
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def load_vad_text_data(en_path, fr_path) -> List[Dict]:
    if not os.path.exists(en_path) or not os.path.exists(fr_path):
        return []
    with open(en_path, "r", encoding="utf-8") as e, open(fr_path, "r", encoding="utf-8") as f:
        en_lines, fr_lines = e.readlines(), f.readlines()

    data = []
    n = min(len(en_lines), len(fr_lines))

    for i in tqdm(range(n), desc=f"Loading {os.path.basename(en_path)}"):
        en = en_lines[i].strip()
        fr = fr_lines[i].strip()

        m_en = re.match(r"^([\w-]+)\s+V:([\d\.-]+)\s+A:([\d\.-]+)\s+D:([\d\.-]+)\s+(.*)$", en)
        m_fr = re.match(r"^([\w-]+)\s+(.*)$", fr)

        if m_en and m_fr and m_en.group(1) == m_fr.group(1):
            try:
                data.append({
                    "AudioID": m_en.group(1),
                    "Valence": float(m_en.group(2)),
                    "Arousal": float(m_en.group(3)),
                    "Dominance": float(m_en.group(4)),
                    "EnglishText": m_en.group(5),
                    "FrenchText": m_fr.group(2)
                })
            except:
                pass
    return data


def load_plain_text_data(en_path, fr_path):
    if not os.path.exists(en_path) or not os.path.exists(fr_path):
        return []
    with open(en_path, "r", encoding="utf-8") as e, open(fr_path, "r", encoding="utf-8") as f:
        en_lines, fr_lines = e.readlines(), f.readlines()

    data = []
    n = min(len(en_lines), len(fr_lines))

    for i in tqdm(range(n), desc=f"Loading {os.path.basename(en_path)}"):
        en = en_lines[i].strip()
        fr = fr_lines[i].strip()

        m_en = re.match(r"^([\w-]+)\s+(.*)$", en)
        m_fr = re.match(r"^([\w-]+)\s+(.*)$", fr)

        if m_en and m_fr and m_en.group(1) == m_fr.group(1):
            data.append({
                "AudioID": m_en.group(1),
                "EnglishText": m_en.group(2),
                "FrenchText": m_fr.group(2)
            })
    return data


def build_prompt(sample, exp_name):
    if exp_name == "baseline":
        return f"English: {sample['EnglishText']}\nFrench:"

    if exp_name == "novel_prompt":
        try:
            v = float(sample["Valence"])
            a = float(sample["Arousal"])
            d = float(sample["Dominance"])
            t = sample["EnglishText"]
            return (
                f"You are a professional translator. The speaker's original tone was "
                f"(Valence: {v:.2f}, Arousal: {a:.2f}, Dominance: {d:.2f}). "
                f"Translate the following English text to French, preserving this emotional context.\n"
                f"English: {t}\nFrench:"
            )
        except:
            return None

    if exp_name == "arousal_eq3":
        try:
            a = float(sample["Arousal"])
            tag = "with" if a >= 0.5 else "without"
            return f"English {tag} arousal: {sample['EnglishText']} \n French:"
        except:
            return None

    return f"English: {sample['EnglishText']}\nFrench:"


def extract_french(text):
    marker = "French:"
    out = text.split(marker, 1)[-1] if marker in text else text
    stops = ["\nEnglish:", "\n\nEnglish:", "\nFrench:", "<eos>"]
    idxs = [out.find(s) for s in stops if s in out]
    if idxs:
        out = out[:min(idxs)]
    return out.strip()


def write_csv(path, rows, fieldnames):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_model_and_tokenizer(adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    base.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def translate_samples(model, tokenizer, samples, exp_name):
    preds = []
    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc=f"Translating ({exp_name})"):
        batch = samples[i:i+BATCH_SIZE]
        prompts = [build_prompt(s, exp_name) for s in batch]

        valid_prompts = [p for p in prompts if p is not None]
        valid_idx = [i for i, p in enumerate(prompts) if p is not None]
        if not valid_prompts:
            preds.extend([None] * len(batch))
            continue

        enc = tokenizer(valid_prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_INPUT_TOKENS).to(model.device)

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
        trans = [extract_french(t) for t in dec]

        batch_preds = [None] * len(batch)
        for j, idx in enumerate(valid_idx):
            batch_preds[idx] = trans[j]

        preds.extend(batch_preds)

    return preds


def process_split(exp, split, en_path, fr_path, model, tokenizer):
    if exp == "baseline":
        samples = load_plain_text_data(en_path, fr_path)
    else:
        samples = load_vad_text_data(en_path, fr_path)

    if not samples:
        return

    translations = translate_samples(model, tokenizer, samples, exp)
    rows = []

    fields = ["id", "AudioID", "EnglishText", "TranslatedFrench", "GroundTruthFrench"]
    if exp != "baseline":
        fields.insert(2, "Dominance")
        fields.insert(2, "Arousal")
        fields.insert(2, "Valence")

    for i, s in enumerate(samples):
        row = {
            "id": i,
            "AudioID": s.get("AudioID", ""),
            "EnglishText": s.get("EnglishText", ""),
            "TranslatedFrench": translations[i],
            "GroundTruthFrench": s.get("FrenchText", "")
        }
        if exp != "baseline":
            row["Valence"] = s.get("Valence", "")
            row["Arousal"] = s.get("Arousal", "")
            row["Dominance"] = s.get("Dominance", "")
        rows.append(row)

    out_csv = os.path.join(OUTPUT_DIR, f"{split}_translations_{exp}.csv")
    write_csv(out_csv, rows, fields)


def main():
    for exp in EXPERIMENTS:
        adapter_dir = os.path.join(PROJECT_DIR, f"towerbase-7b-emotion-{exp}-192")
        if not os.path.exists(adapter_dir):
            continue

        model, tokenizer = load_model_and_tokenizer(adapter_dir)

        if exp == "baseline":
            dev_en, dev_fr = f"{DATA_DIR}/dev_txt/dev_with_ids.en", f"{DATA_DIR}/dev_txt/dev_with_ids.fr"
            test_en, test_fr = f"{DATA_DIR}/test_txt/test_with_ids.en", f"{DATA_DIR}/test_txt/test_with_ids.fr"
        else:
            dev_en, dev_fr = f"{DATA_DIR}/dev_txt/dev_with_ids_vad.en", f"{DATA_DIR}/dev_txt/dev_with_ids.fr"
            test_en, test_fr = f"{DATA_DIR}/test_txt/test_with_ids_vad.en", f"{DATA_DIR}/test_txt/test_with_ids.fr"

        process_split(exp, "dev", dev_en, dev_fr, model, tokenizer)
        process_split(exp, "test", test_en, test_fr, model, tokenizer)

        del model, tokenizer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
