import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.utils import dispatch_for_generation

# Step 1: Load the quantized model and tokenizer
# MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_ID = "TinyLlama-1.1B-Chat-v1.0-W8A8-Dynamic-Per-Token"
# MODEL_ID = "TinyLlama-1.1B-Chat-v1.0-W8A8-Static-Per-Token"
MODEL_ID = "TinyLlama-1.1B-Chat-v1.0-Smooth-GPTQ-W8A8-Dynamic-Per-Token"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


# Step 2: Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
# input_ids = torch.cat([input_ids, input_ids], dim=0)  # Duplicate input for testing
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")
