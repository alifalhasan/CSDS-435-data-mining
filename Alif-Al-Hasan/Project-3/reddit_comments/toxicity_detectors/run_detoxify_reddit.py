import pandas as pd
from detoxify import Detoxify
from tqdm import tqdm


def run_detoxify_reddit(input_path, output_path):
    """
    Run Detoxify on Reddit filtered comments and output toxicity scores (0–10 scale).

    Parameters
    ----------
    input_path : str
        CSV file containing columns: ['comment_id', 'body']
    output_path : str
        Path to save CSV with columns: ['comment_id', 'detoxify_score']
    """
    print(f"\n🔹 Loading comments from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} comments")

    print("🔹 Loading Detoxify model...")
    model = Detoxify("original")

    scores = []
    for text in tqdm(df["body"].tolist(), desc="Detoxify Scoring"):
        try:
            result = model.predict(str(text))
            score = result["toxicity"] * 10  # scale 0–10
        except Exception:
            score = 0.0
        scores.append(score)

    df_out = pd.DataFrame({"comment_id": df["comment_id"], "detoxify_score": scores})
    df_out.to_csv(output_path, index=False)
    print(f"✅ Saved Detoxify scores to: {output_path}")
