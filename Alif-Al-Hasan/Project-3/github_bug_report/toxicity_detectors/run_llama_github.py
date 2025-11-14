import os, csv, torch, gc, time, sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

PROMPT_TEMPLATE = """You are an expert moderator analyzing online discussions.

Answer only "YES" or "NO".

Determine whether the following comment is toxic in a technical context.

Comment:
{content}

Answer:"""


def load_model():
    print(f"\nLoading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tokenizer


def classify(model, tokenizer, text):
    prompt = PROMPT_TEMPLATE.format(content=text[:2000])
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=8, temperature=0.0, do_sample=False
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded[len(prompt) :].strip().upper()
    if "YES" in answer[:5]:
        return "YES"
    elif "NO" in answer[:5]:
        return "NO"
    return "UNCLEAR"


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def detect_toxic_comments(input_path, output_path):
    print(f"\nProcessing: {input_path}")
    with open(input_path, encoding="utf-8", errors="replace") as fin:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames) + ["model_answer", "toxicity_decision"]

        with open(output_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            model, tokenizer = load_model()
            for row in tqdm(reader, desc="Llama classification"):
                text = row.get("body", "")
                if not text.strip():
                    row["model_answer"] = "EMPTY"
                    row["toxicity_decision"] = "UNCLEAR"
                else:
                    ans = classify(model, tokenizer, text)
                    row["model_answer"] = ans
                    row["toxicity_decision"] = "TOXIC" if ans == "YES" else "NON_TOXIC"
                writer.writerow(row)
                free_memory()

    print(f"Saved Llama toxicity results to: {output_path}")


def run_llama_github(input_path, output_path):
    detect_toxic_comments(input_path, output_path)
