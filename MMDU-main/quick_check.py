# quick_check.py
from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq
m = "llava-hf/llava-v1.6-vicuna-7b-hf"
cfg = AutoConfig.from_pretrained(m, trust_remote_code=True)
print("model_type:", cfg.model_type)  # 预期: llava_next
proc = AutoProcessor.from_pretrained(m, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(m, trust_remote_code=True)
print(type(model).__name__)
