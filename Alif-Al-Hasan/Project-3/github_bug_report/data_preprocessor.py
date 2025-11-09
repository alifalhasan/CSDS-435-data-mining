import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def load_and_clean_data(filepath="data/bug_reports.csv"):
    """Load and clean GitHub data"""

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filepath)

    df = pd.read_csv(full_path)

    # Remove very short comments (noise)
    df = df[df["body"].str.len() >= 10].copy()

    # Remove bot comments
    bot_keywords = ["bot", "dependabot", "github-actions", "codecov"]
    df = df[
        ~df["user"]
        .str.lower()
        .str.contains("|".join(bot_keywords), na=False, regex=True)
    ]

    # Parse timestamp
    df["created_at"] = pd.to_datetime(df["created_at"])

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
    print(f"\nAuthor associations:")
    print(df["author_association"].value_counts())

    return df


def exploratory_analysis(df):
    plt.figure(figsize=(10, 4))
    df["author_association"].value_counts().plot(kind="bar")
    plt.xlabel("Author Role")
    plt.ylabel("Count")
    plt.title("Comments by Role")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("eda_basic.png", dpi=150)
    plt.close()

    print("\n=== Basic Statistics ===")
    print(f"Mean comment length: {df['comment_length'].mean():.0f} chars")
    print(
        f"Comments with code: {df['has_code'].sum()} ({df['has_code'].mean()*100:.1f}%)"
    )


def preprocess_github():
    df = load_and_clean_data("data/bug_reports.csv")
    exploratory_analysis(df)

    # Save cleaned data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "data/bug_reports_cleaned.csv")
    df.to_csv(output_path, index=False)
