import os
import torch
import librosa
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from tqdm import tqdm
import json
import logging
import re
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

AUDIO_DIRS = [
    "/data3/ritika/data/raw/train_100h/train/audiofiles",
    "/data3/ritika/data/raw/dev/dev/audiofiles",
    "/data3/ritika/data/raw/test/test/audiofiles"
]

BASE_DATA_DIR = "/data3/ritika/data/raw/"
SPLIT_INFO = {
    "train_100h": {
        "id_file_en": f"{BASE_DATA_DIR}/train_100h_txt/train_with_ids.en",
        "id_file_fr": f"{BASE_DATA_DIR}/train_100h_txt/train_with_ids.fr",
        "output_en": f"{BASE_DATA_DIR}/train_100h_txt/train_with_ids_vad.en",
    },
    "dev": {
        "id_file_en": f"{BASE_DATA_DIR}/dev_txt/dev_with_ids.en",
        "id_file_fr": f"{BASE_DATA_DIR}/dev_txt/dev_with_ids.fr",
        "output_en": f"{BASE_DATA_DIR}/dev_txt/dev_with_ids_vad.en",
    },
    "test": {
        "id_file_en": f"{BASE_DATA_DIR}/test_txt/test_with_ids.en",
        "id_file_fr": f"{BASE_DATA_DIR}/test_txt/test_with_ids.fr",
        "output_en": f"{BASE_DATA_DIR}/test_txt/test_with_ids_vad.en",
    }
}

cache_dir = "/data3/ritika_project/huggingface_cache"
os.environ["HF_HOME"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "models")


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        hidden = self.wav2vec2(input_values)[0]
        pooled = torch.mean(hidden, dim=1)
        logits = self.classifier(pooled)
        return pooled, logits


def load_allowed_ids(split_info_dict):
    allowed = set()
    for _, info in split_info_dict.items():
        for path in [info.get("id_file_en"), info.get("id_file_fr")]:
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        m = re.match(r"^([\w-]+)\s+", line)
                        if m:
                            allowed.add(m.group(1))
    return allowed


logging.info(f"Loading model: {MODEL_ID}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = EmotionModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    exit()


def get_emotion_values(audio_path):
    try:
        speech, sr = librosa.load(audio_path, sr=16000)
        if len(speech) == 0:
            return None

        proc = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = proc["input_values"].to(device)

        with torch.no_grad():
            _, logits = model(input_values)

        arr = logits[0].cpu().numpy()
        return {"valence": float(arr[2]), "arousal": float(arr[0]), "dominance": float(arr[1])}
    except:
        return None


def extract_vad_scores(audio_dirs, allowed_ids):
    results = {}
    paths = []

    for audio_dir in audio_dirs:
        if not os.path.isdir(audio_dir):
            continue
        for subdir, _, files in os.walk(audio_dir):
            for f in files:
                if f.lower().endswith((".wav", ".flac", ".mp3")):
                    full = os.path.join(subdir, f)
                    aid = os.path.splitext(f)[0]
                    if aid in allowed_ids:
                        paths.append(full)

    for audio_path in tqdm(paths, desc="Extracting VAD"):
        vals = get_emotion_values(audio_path)
        if vals:
            aid = os.path.splitext(os.path.basename(audio_path))[0]
            results[aid] = vals

    return results


def combine_vad_with_splits(split_info_dict, vad_results):
    for split, info in split_info_dict.items():
        in_path = info["id_file_en"]
        out_path = info["output_en"]

        if not os.path.exists(in_path):
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                m = re.match(r"^([\w-]+)\s+(.*)", line)
                if m:
                    aid = m.group(1)
                    text = m.group(2).strip()
                    vad = vad_results.get(aid)

                    if vad:
                        fout.write(
                            f"{aid} V:{vad['valence']:.2f} A:{vad['arousal']:.2f} D:{vad['dominance']:.2f} {text}\n"
                        )
                    else:
                        fout.write(line)
                else:
                    fout.write(line)


if __name__ == "__main__":
    allowed_ids = load_allowed_ids(SPLIT_INFO)
    vad_results = extract_vad_scores(AUDIO_DIRS, allowed_ids)
    if vad_results:
        combine_vad_with_splits(SPLIT_INFO, vad_results)
    else:
        logging.error("No VAD results found.")
