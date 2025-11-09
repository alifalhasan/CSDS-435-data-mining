from github_bug_report.data_preprocessor import preprocess_github
from github_bug_report.toxicity_detectors.run_detoxify import run_detoxify
from github_bug_report.toxicity_detectors.run_llama import run_llama

preprocess_github()
run_detoxify()
run_llama()