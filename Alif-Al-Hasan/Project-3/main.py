from github_bug_report.data_preprocessor import preprocess_github
from github_bug_report.toxicity_detectors.run_detoxify_github import run_detoxify_github
from github_bug_report.toxicity_detectors.run_llama_github import run_llama_github
from reddit_comments.data_preprocess import filter_comments_by_post_ids
from reddit_comments.toxicity_detectors.run_detoxify_reddit import run_detoxify_reddit
from reddit_comments.toxicity_detectors.run_llama_reddit import run_llama_reddit
from reddit_comments.toxicity_detectors.combine_toxicity_results import (
    combine_and_filter_toxicity,
)


# GitHub
preprocess_github(
    input_path="github_bug_report/data/bug_reports.csv",
    output_path="github_bug_report/data/bug_reports_cleaned.csv",
)

run_detoxify_github(
    input_path="github_bug_report/data/bug_reports_cleaned.csv",
    output_path="github_bug_report/data/detoxify_results.csv",
)

run_llama_github(
    input_path="github_bug_report/data/bug_reports_cleaned.csv",
    output_path="github_bug_report/data/llama_results.csv",
)


# Reddit
filter_comments_by_post_ids(
    id_file="reddit_comments/data/post_ids.csv",
    comments_file="reddit_comments/data/learnprogramming_comments.jsonl",
    output_file="reddit_comments/data/filtered_comments.csv",
)

run_detoxify_reddit(
    input_path="reddit_comments/data/filtered_comments.csv",
    output_path="reddit_comments/data/reddit_detoxify.csv",
)

run_llama_reddit(
    input_path="reddit_comments/data/filtered_comments.csv",
    output_path="reddit_comments/data/reddit_llama.csv",
)

combine_and_filter_toxicity(
    detoxify_path="reddit_comments/data/reddit_detoxify.csv",
    llama_path="reddit_comments/data/reddit_llama.csv",
    comments_path="reddit_comments/data/filtered_comments.csv",
    output_all="reddit_comments/data/reddit_toxicity_scores.csv",
    output_high="reddit_comments/data/reddit_high_toxic.csv",
)
