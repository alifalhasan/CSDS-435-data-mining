import pandas as pd
from detoxify import Detoxify
from tqdm import tqdm
import numpy as np
import os


def detect_toxicity_detoxify(df, threshold=0.5):
    print("Running Detoxify toxicity detection...")

    model = Detoxify("original")

    results = []
    for comment in tqdm(df["body"].tolist(), desc="Detoxify"):
        try:
            scores = model.predict(comment)
            results.append(
                {
                    "toxicity": scores["toxicity"],
                    "severe_toxicity": scores["severe_toxicity"],
                    "insult": scores["insult"],
                    "is_toxic_detoxify": scores["toxicity"] > threshold,
                }
            )
        except Exception as e:
            results.append(
                {
                    "toxicity": 0.0,
                    "severe_toxicity": 0.0,
                    "insult": 0.0,
                    "is_toxic_detoxify": False,
                }
            )

    results_df = pd.DataFrame(results)

    # Add to original dataframe
    df_result = df.copy()
    df_result["detoxify_score"] = results_df["toxicity"]
    df_result["detoxify_toxic"] = results_df["is_toxic_detoxify"]

    print(f"\nDetoxify complete")
    print(
        f"Toxic comments: {df_result['detoxify_toxic'].sum()} ({df_result['detoxify_toxic'].mean()*100:.1f}%)"
    )

    return df_result


def run_detoxify():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_path = os.path.join(script_dir, "../data/bug_reports_cleaned.csv")
    output_path = os.path.join(script_dir, "../data/processed/detoxify_results.csv")

    df = pd.read_csv(input_path)
    df = detect_toxicity_detoxify(df)
    df.to_csv(output_path, index=False)
