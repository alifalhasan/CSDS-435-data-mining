import pandas as pd
from detoxify import Detoxify
from tqdm import tqdm


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
                    "is_toxic_detoxify": scores["toxicity"] > threshold,
                }
            )
        except Exception:
            results.append({"toxicity": 0.0, "is_toxic_detoxify": False})

    df_result = df.copy()
    df_result["detoxify_score"] = [r["toxicity"] for r in results]
    df_result["detoxify_toxic"] = [r["is_toxic_detoxify"] for r in results]

    print(f"✅ Detoxify complete. Toxic comments: {df_result['detoxify_toxic'].sum()}")
    return df_result


def run_detoxify_github(input_path, output_path):
    df = pd.read_csv(input_path)
    df = detect_toxicity_detoxify(df)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved Detoxify results to: {output_path}")
