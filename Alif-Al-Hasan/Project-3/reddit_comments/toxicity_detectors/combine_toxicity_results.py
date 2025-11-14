import pandas as pd


def combine_and_filter_toxicity(
    detoxify_path, llama_path, comments_path, output_all, output_high
):
    """
    Combine Detoxify + LLaMA scores, and filter comments where both >= 7.
    """
    print("\nCombining model outputs...")

    df_detox = pd.read_csv(detoxify_path)
    df_llama = pd.read_csv(llama_path)
    df_comments = pd.read_csv(comments_path)

    df_combined = df_comments.merge(df_detox, on="comment_id", how="left").merge(
        df_llama, on="comment_id", how="left"
    )
    df_combined.to_csv(output_all, index=False)
    print(f"Saved all toxicity scores to: {output_all}")

    # Filter where both models gave score >= 7
    df_high = df_combined[
        (df_combined["detoxify_score"] >= 7) | (df_combined["llama_score"] >= 7)
    ]
    df_high.to_csv(output_high, index=False)
    print(f"Saved high-toxicity comments to: {output_high}")
    print(f"Found {len(df_high):,} high-toxicity comments")
