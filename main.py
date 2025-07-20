import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.utils import dispatch_for_generation


def quantize(
    model_id: str,
    scheme: str = "SYM",
    method: str = "RTN",
    num_calibration_samples: int = 512,
    max_sequence_length: int = 2048,
):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load and preprocess dataset
    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"
    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{num_calibration_samples}]")
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)

    def tokenize_fn(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names)

    # Choose quantization scheme
    quant_scheme = "W8A8" if scheme == "SYM" else "W8A8_ASYM"

    # Choose quantization method and build recipe
    if method == "RTN":
        recipe = [
            QuantizationModifier(targets="Linear", scheme=quant_scheme, ignore=["lm_head"]),
        ]
    elif method == "Smooth-GPTQ":
        recipe = [
            SmoothQuantModifier(smoothing_strength=0.8),
            GPTQModifier(targets="Linear", scheme=quant_scheme, ignore=["lm_head"]),
        ]
    else:
        raise ValueError("Unknown quantization method.")

    # Apply quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_sequence_length,
        num_calibration_samples=num_calibration_samples,
    )

    # Confirm generations of the quantized model look sane.
    print("========== SAMPLE GENERATION ==============")
    dispatch_for_generation(model)
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids, max_new_tokens=10)
    print(tokenizer.decode(output[0]))
    print("==========================================\n\n")

    # Save the compressed model
    print("========== SAVING COMPRESSED MODEL ==============")
    save_dir = model_id.rstrip("/").split("/")[-1] + f"-test-{scheme}-{method}-W8A8-Dynamic-Per-Token"
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    print(f"Compressed model saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="LLM Compressor Quantization Example")
    parser.add_argument("--model_id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model ID to use")
    parser.add_argument("--scheme", type=str, choices=["SYM", "ASYM"], default="SYM", help="Quantization scheme: SYM or ASYM")
    parser.add_argument("--method", type=str, choices=["RTN", "Smooth-GPTQ"], default="Smooth-GPTQ", help="Quantization method: RTN or SmoothQuant + GPTQ")
    parser.add_argument("--num_calibration_samples", type=int, default=512, help="Number of calibration samples")
    parser.add_argument("--max_sequence_length", type=int, default=2048, help="Max sequence length")
    args = parser.parse_args()

    quantize(
        model_id=args.model_id,
        scheme=args.scheme,
        method=args.method,
        num_calibration_samples=args.num_calibration_samples,
        max_sequence_length=args.max_sequence_length,
    )


if __name__ == "__main__":
    main()