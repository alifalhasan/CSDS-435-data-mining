# Cross-Platform Toxicity Analysis in Programming Support Communities

## Overview

This project investigates toxicity patterns across two major programming support platforms: **r/learnprogramming** (Reddit) and **GitHub bug report discussions**. We compare two automated toxicity detection methods—**Detoxify** (pre-trained BERT) and **Llama-3.3-70B** (LLM with custom prompts)—to understand how platform context shapes toxic interactions.

## Project Structure

```
PROJECT-3/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── main.py                            # Main execution script
│
├── github_bug_report/                 # GitHub analysis pipeline
│   ├── data/
│   │   ├── ground_truths.csv         # 80 manually labeled toxic comments
│   │   └── results.csv               # Detection results
│   └── toxicity_detectors/
│       ├── run_detoxify_github.py    # Detoxify on GitHub data
│       ├── run_llama_github.py       # Llama on GitHub data
│       └── data_preprocessor.py      # GitHub data preprocessing
│
└── reddit_comments/                   # Reddit analysis pipeline
    ├── data/
    │   ├── filtered_comments.csv     # Preprocessed Reddit comments
    │   ├── reddit_detoxify.csv       # Detoxify results
    │   ├── reddit_high_toxic.csv     # High toxicity comments (score ≥7)
    │   ├── reddit_llama.csv          # Llama results
    │   └── reddit_toxicity_scores.csv # Combined scores
    └── toxicity_detectors/
        ├── combine_toxicity_result.py # Merge detection results
        ├── run_detoxify_reddit.py    # Detoxify on Reddit data
        ├── run_llama_reddit.py       # Llama on Reddit data
        └── data_preprocess.py        # Reddit data preprocessing
```

---

## Datasets

### Dataset 1: Reddit r/learnprogramming

- **Source**: 1,500 posts from r/learnprogramming (Jan 2023 - Nov 2024) containing learning-centered emotions (LCEs)
- **Size**: 16,266 comments
- **Context**: Novice programmers seeking help, often expressing frustration, confusion, or anxiety
- **Emotions**: Anxiety, Boredom, Confusion, Curiosity, Delight, Engagement, Frustration, Surprise

### Dataset 2: GitHub Bug Report Discussions

- **Source**: 100 active repositories (1,000+ stars) from 2024
- **Size**: 91,929 comments (sampled 9,000 for analysis)
- **Context**: Technical bug diagnosis and resolution
- **Metadata**: User roles (OWNER, CONTRIBUTOR, MEMBER, etc.), timestamps, issue URLs
- **Ground Truth**: 80 manually labeled toxic comments

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for Llama)
- Hugging Face account (for Llama model access)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/alifalhasan/CSDS-435-data-mining.git
cd CSDS-435-data-mining/Alif-Al-Hasan/Project-3
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download datasets**
   - Reddit data: Place `filtered_comments.csv` in `reddit_comments/data/`
   - GitHub data: Place dataset in `github_bug_report/data/`

4. **Configure Llama access** (if using Llama)
   - Get Hugging Face token: https://huggingface.co/settings/tokens
   - Set environment variable:
     ```bash
     export HF_TOKEN="your_huggingface_token"
     ```

---

## Usage

### Reddit Analysis Pipeline

#### 1. Preprocess Reddit Data
```bash
cd reddit_comments/toxicity_detectors
python data_preprocess.py
```
**Input**: Raw Reddit comments  
**Output**: `filtered_comments.csv` (cleaned, linked to post emotions)

#### 2. Run Detoxify Detection
```bash
python run_detoxify_reddit.py
```
**Input**: `filtered_comments.csv`  
**Output**: `reddit_detoxify.csv` (toxicity scores 0-1)

#### 3. Run Llama Detection
```bash
python run_llama_reddit.py
```
**Input**: `filtered_comments.csv`  
**Output**: `reddit_llama.csv` (toxicity scores 0-10)

#### 4. Combine Results
```bash
python combine_toxicity_result.py
```
**Input**: `reddit_detoxify.csv`, `reddit_llama.csv`  
**Output**: 
- `reddit_toxicity_scores.csv` (all scores)
- `reddit_high_toxic.csv` (comments with score ≥7)

#### 5. Run Full Pipeline
```bash
python main.py
```
Executes steps 1-4 sequentially.

---

### GitHub Analysis Pipeline

#### 1. Preprocess GitHub Data
```bash
cd github_bug_report/toxicity_detectors
python data_preprocessor.py
```
**Input**: Raw GitHub issue comments  
**Output**: Cleaned dataset with bot removal, length filtering

#### 2. Run Detoxify Detection
```bash
python run_detoxify_github.py
```
**Input**: Preprocessed GitHub comments  
**Output**: Detoxify toxicity scores

#### 3. Run Llama Detection
```bash
python run_llama_github.py
```
**Input**: Preprocessed GitHub comments  
**Output**: Llama toxicity scores (0-10)

#### 4. Evaluate Against Ground Truth
Compare results with `ground_truths.csv` (80 manually labeled toxic comments).

---

## Methodology

### Detection Methods

#### Method 1: Detoxify
- **Type**: Pre-trained BERT-based classifier
- **Training**: Social media toxicity data
- **Output**: Probability score (0-1)
- **Threshold**: 0.5 for binary classification

#### Method 2: Llama-3.3-70B-Instruct
- **Type**: Large language model with custom prompts
- **Approach**: Context-aware scoring with platform-specific definitions
- **Output**: Toxicity score (0-10)
  - 0-3: Neutral to slightly dismissive
  - 4-6: Unhelpful, condescending
  - 7-8: Rude, mocking, gatekeeping
  - 9-10: Extremely toxic
- **Threshold**: 7 for binary classification

### Evaluation Metrics

**GitHub Ground Truth (80 toxic comments)**:
- Detoxify: 18/80 detected (22.5% recall)
- Llama: 58/80 detected (72.5% recall)
- **Performance Gap**: 3.2× improvement with Llama

---

## References

- **GitHub Dataset**: [Silent Is Not Actually Silent: An Investigation of Toxicity on Bug Report Discussion](https://dl.acm.org/doi/abs/10.1145/3696630.3728502)
- **Reddit Dataset**: From an ongoing research.
- **Detoxify**: [unitaryai/detoxify](https://github.com/unitaryai/detoxify)
- **Llama**: [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)