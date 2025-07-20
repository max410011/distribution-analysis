# distribution-analysis

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e llm-compressor
pip install -e compressed-tensors
```
## Quantize the model
```bash
python tinyllama_example.py
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