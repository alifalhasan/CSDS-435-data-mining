import os
import pandas as pd
import matplotlib.pyplot as plt


def load_and_clean_data(input_path):
    """Load and clean GitHub data"""
    df = pd.read_csv(input_path)

    df = df[df["body"].str.len() >= 10].copy()  # Remove very short comments

    # Remove bot comments
    bot_keywords = ["bot", "dependabot", "github-actions", "codecov"]
    df = df[
        ~df["user"]
        .str.lower()
        .str.contains("|".join(bot_keywords), na=False, regex=True)
    ]

    # Parse timestamp
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # Extract repo name
    df["repo"] = df["issue_url"].str.extract(
        r"github\.com/([^/]+/[^/]+)/issues", expand=False
    )

    # Feature engineering
    df["comment_length"] = df["body"].str.len()
    df["word_count"] = df["body"].str.split().str.len()
    df["has_code"] = df["body"].str.contains("```", regex=False)
    df["has_link"] = df["body"].str.contains("http", regex=False)
    df["has_mention"] = df["body"].str.contains("@", regex=False)

    print(f"After cleaning: {len(df)} comments")
    print(f"Repositories: {df['repo'].nunique()}")

    return df


def exploratory_analysis(df, output_dir="github_bug_report/data"):
    plt.figure(figsize=(10, 4))
    df["author_association"].value_counts().plot(kind="bar")
    plt.xlabel("Author Role")
    plt.ylabel("Count")
    plt.title("Comments by Role")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_basic.png"), dpi=150)
    plt.close()


def preprocess_github(input_path, output_path):
    df = load_and_clean_data(input_path)
    exploratory_analysis(df, os.path.dirname(output_path))
    df.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned dataset to: {output_path}")
