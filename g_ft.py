import os
import re
import logging
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

PROJECT_DIR = '/data3/ritika_project/ml'
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
CACHE_DIR = os.path.join(PROJECT_DIR, 'huggingface_cache')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'gemma_output')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(CACHE_DIR, 'datasets')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(CACHE_DIR, 'models')

logging.basicConfig(level=logging.INFO)
logging.info("Starting GEMMA fine-tuning...")

EXPERIMENTS_TO_RUN = ['baseline','arousal_eq3','novel_prompt']
BASE_MODEL = 'google/gemma-7b'

# load model
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
tokenizer.padding_side = "right"

# prepare LoRA
try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
except:
    model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# simple loader for VAD datasets
def load_vad_data(en_path, fr_path):
    if not os.path.exists(en_path) or not os.path.exists(fr_path):
        logging.error("Missing VAD files")
        return None

    data = []
    with open(en_path, 'r', encoding='utf-8') as f_en, open(fr_path, 'r', encoding='utf-8') as f_fr:
        en_lines, fr_lines = f_en.readlines(), f_fr.readlines()

        for i in tqdm(range(min(len(en_lines), len(fr_lines))), desc=os.path.basename(en_path)):
            e = en_lines[i].strip()
            f = fr_lines[i].strip()

            m_en = re.match(r'^([\w-]+)\s+V:([\d\.-]+)\s+A:([\d\.-]+)\s+D:([\d\.-]+)\s+(.*)$', e)
            m_fr = re.match(r'^([\w-]+)\s+(.*)$', f)

            if m_en and m_fr and m_en.group(1) == m_fr.group(1):
                data.append({
                    "AudioID": m_en.group(1),
                    "Valence": float(m_en.group(2)),
                    "Arousal": float(m_en.group(3)),
                    "Dominance": float(m_en.group(4)),
                    "EnglishText": m_en.group(5),
                    "FrenchText": m_fr.group(2)
                })
    return data

# simple loader for plain text datasets
def load_plain_data(en_path, fr_path):
    if not os.path.exists(en_path) or not os.path.exists(fr_path):
        logging.error("Missing plain text files")
        return None

    data = []
    with open(en_path, 'r', encoding='utf-8') as f_en, open(fr_path, 'r', encoding='utf-8') as f_fr:
        en_lines, fr_lines = f_en.readlines(), f_fr.readlines()

        for i in tqdm(range(min(len(en_lines), len(fr_lines))), desc=os.path.basename(en_path)):
            e = en_lines[i].strip()
            f = fr_lines[i].strip()

            m_en = re.match(r'^([\w-]+)\s+(.*)$', e)
            m_fr = re.match(r'^([\w-]+)\s+(.*)$', f)

            if m_en and m_fr and m_en.group(1) == m_fr.group(1):
                data.append({
                    "AudioID": m_en.group(1),
                    "EnglishText": m_en.group(2),
                    "FrenchText": m_fr.group(2)
                })
    return data

# prompt formats
def fmt_baseline(s):
    return f"English: {s['EnglishText']}\nFrench: {s['FrenchText']}"

def fmt_arousal(s):
    status = "with" if float(s['Arousal']) >= 0.5 else "without"
    return f"English {status} arousal: {s['EnglishText']}\nFrench: {s['FrenchText']}"

def fmt_novel(s):
    return (
        f"You are a professional translator. (V:{s['Valence']:.2f}, A:{s['Arousal']:.2f}, D:{s['Dominance']:.2f})\n"
        f"English: {s['EnglishText']}\nFrench: {s['FrenchText']}"
    )

# paths
train_vad_en = f"{DATA_DIR}/train_100h_txt/train_with_ids_vad.en"
train_vad_fr = f"{DATA_DIR}/train_100h_txt/train_with_ids_vad.fr"
dev_vad_en   = f"{DATA_DIR}/dev_txt/dev_with_ids_vad.en"
dev_vad_fr   = f"{DATA_DIR}/dev_txt/dev_with_ids.fr"

train_plain_en = f"{DATA_DIR}/train_100h_txt/train_with_ids.en"
train_plain_fr = f"{DATA_DIR}/train_100h_txt/train_with_ids.fr"
dev_plain_en   = f"{DATA_DIR}/dev_txt/dev_with_ids.en"
dev_plain_fr   = f"{DATA_DIR}/dev_txt/dev_with_ids.fr"

# training loop
for exp in EXPERIMENTS_TO_RUN:
    logging.info(f"Running experiment: {exp}")

    if exp == 'baseline':
        loader = load_plain_data
        fmt = fmt_baseline
        tr_en, tr_fr = train_plain_en, train_plain_fr
        dv_en, dv_fr = dev_plain_en, dev_plain_fr
    else:
        loader = load_vad_data
        fmt = fmt_arousal if exp == 'arousal_eq3' else fmt_novel
        tr_en, tr_fr = train_vad_en, train_vad_fr
        dv_en, dv_fr = dev_vad_en, dev_vad_fr

    train_raw = loader(tr_en, tr_fr)
    val_raw = loader(dv_en, dv_fr)
    if not train_raw or not val_raw:
        logging.error(f"Data load failed for {exp}")
        continue

    train_ds = Dataset.from_list(train_raw)
    val_ds   = Dataset.from_list(val_raw)

    train_ds = train_ds.map(lambda s: {"text": fmt(s)}, remove_columns=train_ds.column_names)\
                       .filter(lambda x: x['text'] is not None)

    val_ds = val_ds.map(lambda s: {"text": fmt(s)}, remove_columns=val_ds.column_names)\
                   .filter(lambda x: x['text'] is not None)

    model_lora = get_peft_model(model, lora_cfg)

    args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, f"results_gemma_{exp}"),
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_ratio=0.03,
        num_train_epochs=3,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=250,
        save_strategy="epoch",
        logging_steps=20,
        report_to="tensorboard"
    )

    trainer = SFTTrainer(
        model=model_lora,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=192,
        args=args,
        packing=False
    )

    trainer.train()
    model_lora.save_pretrained(os.path.join(OUTPUT_DIR, f"gemma-7b-emotion-{exp}-192"))

logging.info("All experiments complete.")
