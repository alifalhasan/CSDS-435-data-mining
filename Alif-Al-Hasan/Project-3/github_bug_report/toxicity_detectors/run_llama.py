#!/usr/bin/env python3
"""
Detects toxic or hostile comments using Llama 3.3-70B-Instruct.
"""

import os, csv, torch, gc, time, sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

PROMPT_TEMPLATE = """You are an expert moderator analyzing online discussions.

Answer only "YES" or "NO".

Determine whether the following comment is toxic in a technical context.
Toxicity includes:
- hostile, insulting, or condescending tone
- dismissive remarks like "RTFM", "duplicate", "won't fix"
- profanity, rudeness, or blame
- gatekeeping language ("you shouldn't use this")

Comment:
{content}

Answer:"""

HF_CACHE = f"/tmp/hf_models_{os.getenv('USER', 'default_user')}"
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.makedirs(HF_CACHE, exist_ok=True)


def load_model():
    """Load Llama model and tokenizer."""
    print(f"\n🔹 Loading {MODEL_NAME} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load {MODEL_NAME}: {e}")
        return None, None


def classify(model, tokenizer, text):
    """Classify one comment as toxic or not."""
    if not model or not tokenizer:
        return "ERROR"

    prompt = PROMPT_TEMPLATE.format(content=text[:2000])
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(model.device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=8, temperature=0.0, do_sample=False
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = decoded[len(prompt) :].strip().upper()
        if "YES" in answer[:5]:
            return "YES"
        elif "NO" in answer[:5]:
            return "NO"
        else:
            return "UNCLEAR"
    except Exception as e:
        print(f"Classification failed: {e}")
        return "ERROR"


def free_memory():
    """Free GPU + CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def count_rows(csv_path):
    """Count total rows in CSV without loading into memory."""
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10**7)

    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def detect_toxic_comments(csv_path):
    """Detect toxic comments and save results (streaming mode for large files)."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_csv_path = os.path.join(script_dir, csv_path)

    # Increase CSV field size limit
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(10**7)

    print(f"\nCounting rows in {csv_path}...")
    try:
        row_count = count_rows(full_csv_path)
        print(f"Found {row_count:,} comments to process")
    except Exception as e:
        print(f"Could not count rows: {e}")
        row_count = None

    # Load model once
    model, tokenizer = load_model()
    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Prepare output file
    out_file = full_csv_path.replace(".csv", "_llama_filtered.csv")

    # Check if partial results exist (for resume capability)
    processed_count = 0
    if os.path.exists(out_file):
        response = input(
            f"\nOutput file already exists: {out_file}\nOverwrite? (y/n): "
        )
        if response.lower() != "y":
            print("Cancelled.")
            return

    toxic_count = 0
    non_toxic_count = 0
    unclear_count = 0
    error_count = 0

    print(f"\nStarting processing...")

    try:
        with open(full_csv_path, encoding="utf-8", errors="replace") as fin:
            reader = csv.DictReader(fin)

            # Add new columns to fieldnames
            fieldnames = list(reader.fieldnames)
            if "model_answer" not in fieldnames:
                fieldnames.append("model_answer")
            if "toxicity_decision" not in fieldnames:
                fieldnames.append("toxicity_decision")

            with open(out_file, "w", newline="", encoding="utf-8") as fout:
                writer = csv.DictWriter(fout, fieldnames=fieldnames)
                writer.writeheader()

                # Process with progress bar
                pbar = tqdm(reader, total=row_count, desc="Processing", unit="rows")

                for row in pbar:
                    processed_count += 1

                    # Get comment text
                    text = (
                        row.get("comment") or row.get("body") or row.get("text") or ""
                    )

                    # Skip empty comments
                    if not text.strip():
                        row["model_answer"] = "EMPTY"
                        row["toxicity_decision"] = "UNCLEAR"
                        writer.writerow(row)
                        unclear_count += 1
                        continue

                    # Classify
                    vote = classify(model, tokenizer, text)
                    row["model_answer"] = vote

                    # Determine toxicity
                    if vote == "YES":
                        row["toxicity_decision"] = "TOXIC"
                        toxic_count += 1
                    elif vote == "NO":
                        row["toxicity_decision"] = "NON_TOXIC"
                        non_toxic_count += 1
                    elif vote == "ERROR":
                        row["toxicity_decision"] = "ERROR"
                        error_count += 1
                    else:
                        row["toxicity_decision"] = "UNCLEAR"
                        unclear_count += 1

                    # Write immediately (streaming)
                    writer.writerow(row)

                    # Update progress bar
                    pbar.set_postfix(
                        {
                            "toxic": toxic_count,
                            "non-toxic": non_toxic_count,
                            "unclear": unclear_count,
                            "errors": error_count,
                        }
                    )

                    # Small delay to prevent overwhelming GPU
                    time.sleep(0.05)

                    # Periodic memory cleanup (every 100 rows)
                    if processed_count % 100 == 0:
                        free_memory()

                pbar.close()

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Processed {processed_count} rows so far.")
        print(f"Partial results saved to: {out_file}")
    except Exception as e:
        print(f"\n\nError during processing: {e}")
        print(f"Processed {processed_count} rows before error.")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up model
        del model, tokenizer
        free_memory()

    # Print final statistics
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Output file: {out_file}")
    print(f"Total processed: {processed_count:,} rows")
    print(f"")
    print(f"Toxic:       {toxic_count:,} ({toxic_count/processed_count*100:.1f}%)")
    print(
        f"Non-toxic:   {non_toxic_count:,} ({non_toxic_count/processed_count*100:.1f}%)"
    )
    print(f"Unclear:     {unclear_count:,} ({unclear_count/processed_count*100:.1f}%)")
    print(f"Errors:      {error_count:,} ({error_count/processed_count*100:.1f}%)")
    print(f"{'='*60}\n")


def run_llama():
    """Main entry point."""
    detect_toxic_comments("../data/bug_reports_cleaned.csv")


if __name__ == "__main__":
    run_llama()
