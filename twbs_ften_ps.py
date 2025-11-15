import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import os
import pandas as pd
import logging
import re
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cache_dir = "/data3/ritika_project/ml/huggingface_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'models')

EXPERIMENTS_TO_RUN = ['novel_prompt']

base_model_path = "/data3/ritika_project/ml/TowerBase-7B-v0.1"
output_base_dir = "/data3/ritika_project/ml/e2f_w_em"
data_dir = "/data3/ritika_project/ml/data"

os.makedirs(output_base_dir, exist_ok=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
prepared_base_model = prepare_model_for_kbit_training(base_model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


def load_vad_text_data(en_path, fr_path):
    if not os.path.exists(en_path) or not os.path.exists(fr_path):
        return None

    data = []
    with open(en_path, 'r', encoding='utf-8') as e, open(fr_path, 'r', encoding='utf-8') as f:
        en_lines = e.readlines()
        fr_lines = f.readlines()

    n = min(len(en_lines), len(fr_lines))

    for i in tqdm(range(n), desc=f"Loading {os.path.basename(en_path)}"):
        en = en_lines[i].strip()
        fr = fr_lines[i].strip()

        m_en = re.match(r'^([\w-]+)\s+V:([\d\.-]+)\s+A:([\d\.-]+)\s+D:([\d\.-]+)\s+(.*)$', en)
        m_fr = re.match(r'^([\w-]+)\s+(.*)$', fr)

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


def create_prompt_format_arousal_eq3(sample):
    try:
        status = "with" if float(sample['Arousal']) >= 0.5 else "without"
        return f"English {status} arousal: {sample['EnglishText']} \\n French: {sample['FrenchText']}"
    except:
        return None


def create_prompt_format_arousal_eq4(sample):
    try:
        status = "with" if float(sample['Arousal']) >= 0.5 else "without"
        return f"English: {sample['EnglishText']} \\n French {status} arousal: {sample['FrenchText']}"
    except:
        return None


def create_prompt_format_dominance_eq4(sample):
    try:
        status = "with" if float(sample['Dominance']) >= 0.5 else "without"
        return f"English: {sample['EnglishText']} \\n French {status} dominance: {sample['FrenchText']}"
    except:
        return None


def create_prompt_format_novel(sample):
    try:
        v, a, d = float(sample['Valence']), float(sample['Arousal']), float(sample['Dominance'])
        return (
            f"You are a professional translator. "
            f"The speaker's original tone was (Valence: {v:.2f}, Arousal: {a:.2f}, Dominance: {d:.2f}). "
            f"Translate the following English text to French, preserving this emotional context.\\n"
            f"English: {sample['EnglishText']}\\nFrench: {sample['FrenchText']}"
        )
    except:
        return None


train_en = os.path.join(data_dir, "train_100h_txt", "train_with_ids_vad.en")
train_fr = os.path.join(data_dir, "train_100h_txt", "train_with_ids.fr")
dev_en = os.path.join(data_dir, "dev_txt", "dev_with_ids_vad.en")
dev_fr = os.path.join(data_dir, "dev_txt", "dev_with_ids.fr")

train_raw = load_vad_text_data(train_en, train_fr)
val_raw = load_vad_text_data(dev_en, dev_fr)

train_dataset_raw = Dataset.from_list(train_raw)
val_dataset_raw = Dataset.from_list(val_raw)

del train_raw, val_raw


for exp in EXPERIMENTS_TO_RUN:
    if exp == 'arousal_eq3':
        prompt_fn = create_prompt_format_arousal_eq3
    elif exp == 'novel_prompt':
        prompt_fn = create_prompt_format_novel
    else:
        continue

    formatted_train = train_dataset_raw.map(lambda x: {'text': prompt_fn(x)}, remove_columns=train_dataset_raw.column_names)
    formatted_val = val_dataset_raw.map(lambda x: {'text': prompt_fn(x)}, remove_columns=val_dataset_raw.column_names)

    formatted_train = formatted_train.filter(lambda x: x['text'] is not None)
    formatted_val = formatted_val.filter(lambda x: x['text'] is not None)

    model = get_peft_model(prepared_base_model, lora_config)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_base_dir, f"results_{exp}"),
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        optim="adamw_8bit",
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_train_epochs=3,
        bf16=True,
        fp16=False,
        evaluation_strategy="steps",
        eval_steps=250,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        logging_steps=20,
        max_grad_norm=0.3,
        weight_decay=0.001,
        group_by_length=True,
        report_to="tensorboard",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_train,
        eval_dataset=formatted_val,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=192,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    res = trainer.train(resume_from_checkpoint=True)

    final_path = os.path.join(output_base_dir, f"towerbase-7b-emotion-{exp}-192")
    trainer.model.save_pretrained(final_path)

    del model, trainer, formatted_train, formatted_val, training_args
    torch.cuda.empty_cache()

logging.info("All experiments complete.")
