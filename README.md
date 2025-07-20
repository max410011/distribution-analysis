# distribution-analysis


## Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or, if you prefer pip
pip install uv
```

## Install
```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install -e llm-compressor
uv pip install -e compressed-tensors
```

## Quantize the model
```bash
uv run tinyllama_example.py
```
- `--model_id`  
  Specify the HuggingFace model ID (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0).

- `--scheme`  
  Quantization scheme:  
  `SYM` = symmetric quantization (default)  
  `ASYM` = asymmetric quantization

- `--method`  
  Quantization method:  
  `RTN` = Round-To-Nearest (default)  
  `Smooth-GPTQ` = SmoothQuant + GPTQ

- `--num_calibration_samples`  
  Number of calibration samples (default: 512).

- `--max_sequence_length`  
  Maximum sequence length for calibration (default: 2048).