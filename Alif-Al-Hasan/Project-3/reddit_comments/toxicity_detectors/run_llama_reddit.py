import os, torch, gc, time
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

PROMPT_TEMPLATE = """You are an expert content moderator. You are evaluating the tone of online technical discussion comments, especially in forums where beginner programmers ask for help.\n\n"
Your task is to assign a toxicity score between 0 and 10 to each comment based on how negative, discouraging, or hostile it might feel to a novice.\n\n
    "Guidelines:\n"
    "- 0 = perfectly kind, neutral, or encouraging.\n"
    "- 1–3 = slightly dismissive or impatient but still mostly acceptable.\n"
    "- 4–6 = unhelpful, condescending, or somewhat harsh; could discourage a beginner.\n"
    "- 7–8 = clearly rude, mocking, or elitist; shows hostility or gatekeeping.\n"
    "- 9–10 = extremely toxic, hateful, or explicitly offensive.\n\n"
    "Consider factors like:\n"
    "- Condescension, mockery, or superiority toward the question asker.\n"
    "- Hostility, blame, or sarcasm.\n"
    "- Dismissiveness ('just Google it', 'this is stupid', etc.).\n"
    "- Personal attacks or profanity.\n"
    "- Overall tone appropriateness for a beginner-friendly community.\n\n"
    "Output only a single number between 0 and 10 — no explanation or extra text."

Comment:
{content}

Score:"""


def run_llama_reddit(input_path, output_path):
    """
    Runs LLaMA toxicity scoring on Reddit comments (0–10 scale).
    """
    print(f"\nLoading comments from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} comments")

    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    def score_comment(comment):
        prompt = PROMPT_TEMPLATE.format(content=str(comment)[:1500])
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=16, temperature=0.0)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded[len(prompt) :].strip()
        try:
            num = float(response.split()[0])
            return min(max(num, 0), 10)  # clamp between 0–10
        except Exception:
            return 0.0

    llama_scores = []
    for text in tqdm(df["body"].tolist(), desc="LLaMA Scoring"):
        try:
            score = score_comment(text)
        except Exception:
            score = 0.0
        llama_scores.append(score)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.05)

    df_out = pd.DataFrame({"comment_id": df["comment_id"], "llama_score": llama_scores})
    df_out.to_csv(output_path, index=False)
    print(f"Saved LLaMA scores to: {output_path}")
