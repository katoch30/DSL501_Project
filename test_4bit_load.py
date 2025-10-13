# test_4bit_load.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "mistralai/Mistral-7B-v0.1"  # or another small test model if you prefer

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("torch.cuda:", torch.cuda.is_available(), "cuda_ver:", torch.version.cuda)
print("Attempting to load model in 4-bit (this downloads weights)...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
print("Model loaded. Checking trainable flags...")
print("is_quantized:", getattr(model, "is_quantized", None))
