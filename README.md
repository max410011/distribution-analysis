# distribution-analysis

## Clone the repo
```bash
git clone --recursive https://github.com/max410011/distribution-analysis.git
```
You will see the `main.py` and 2 submodule (llm-compressor and compressed-tensors)

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
## Quantize the model
```bash
python main.py --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --scheme SYM --method Smooth-GPTQ
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