# distribution-analysis

A toolkit for quantizing HuggingFace LLMs and analyzing quantized tensor distributions.

## Clone the Repository
```bash
git clone --recursive https://github.com/max410011/distribution-analysis.git
```
You will see the following structure:

- `main.py`
- `llm-compressor`: Algorithms for LLM quantization and compression.
- `compressed-tensors`:  Efficient operations for compressed tensors.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e llm-compressor
pip install -e compressed-tensors
```
## Quantize the model
Run the following command to quantize a model:
```bash
python main.py --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --scheme SYM --method Smooth-GPTQ
```
Arguments:
- `--model_id`  
  HuggingFace model ID (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)

- `--scheme`  
  Quantization scheme:  
  `SYM` = symmetric quantization (default)  
  `ASYM` = asymmetric quantization

- `--method`  
  Quantization method:  
  `RTN` = Round-To-Nearest
  `Smooth-GPTQ` = SmoothQuant + GPTQ (default)  

- `--num_calibration_samples`  
  Number of calibration sampless (default: 512).

- `--max_sequence_length`  
  Maximum sequence length for model calibration (default: 2048).

## Analysis the quantized model
You can run `analysis.ipynb` to visualize and analyze the distributions of quantized weights and activations.  
The notebook will generate figures and CSV results in the `figures/` and `output/` directories for further inspection.