# Explainable Suicide Risk Detection — Transformer Pipeline

A multi-task NLP pipeline for detecting **suicide-risk level**, identifying associated **risk/protective factors**, and extracting **text evidence spans** from social-media posts.

> **Important:** This project is intended for research/educational use and competition experimentation. It is **not a clinical diagnostic tool**, crisis-response system, or substitute for assessment by qualified mental-health professionals. Predictions should not be used alone to make decisions about a person's safety or treatment.

## Overview

The project uses a transformer-based multi-task architecture built around `mental/mental-roberta-base`. A single encoder supports three related tasks:

1. **Risk-level classification** — predicts one of:
   - `Indicator`
   - `Ideation`
   - `Behavior`
   - `Attempt`

2. **Factor classification** — predicts multiple applicable categories from a 24-category taxonomy, including risk and protective factors.

3. **Evidence extraction** — identifies the portions of the post that support the predicted suicide-risk level.

The notebook also combines multiple modeling signals for factor prediction:

- Transformer-based factor probabilities
- TF-IDF + Logistic Regression
- Optional zero-shot LLM factor classification using `microsoft/Phi-3.5-mini-instruct`

The final predictions are produced using cross-validation ensembles and pooled out-of-fold calibration.

## Model Architecture

```text
                    Input Social-Media Post
                              │
                              ▼
                 mental/mental-roberta-base
                       Shared Encoder
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
        Risk-Level Head   Factors Head    Evidence Head
              │               │                │
       ┌──────┴──────┐        │          Token-level
       │             │        │          classification
       ▼             ▼        ▼
   Standard       CORAL    24-category
   Classifier     Ordinal   Multi-label
                  Head       Prediction
       │             │
       └──────┬──────┘
              ▼
       Risk Probability
          Ensemble
```

For risk prediction, the standard classifier and CORAL ordinal head are blended.

For factor prediction, the notebook can blend:

```text
Transformer + TF-IDF/Logistic Regression + optional LLM signal
```

## Key Features

- Domain-adapted RoBERTa backbone
- Multi-task learning
- 4-level ordinal suicide-risk classification
- 24-category multi-label factor prediction
- Token-level evidence extraction
- Character-offset-based evidence alignment
- Rare-factor oversampling
- Alpha-weighted focal loss for factor imbalance
- Pooled out-of-fold threshold tuning
- Risk-head ensemble tuning
- Neural/classical factor-model blending
- Optional zero-shot LLM factor signal
- Grouped stratified cross-validation by `anon_user_id`
- Error analysis with confusion matrix and per-category F1
- Automatic CSV submission generation

## Risk Levels

The notebook uses the following ordered risk levels:

| Level | Description |
|---|---|
| `Indicator` | Lowest level in the competition taxonomy |
| `Ideation` | Suicidal thoughts/ideation |
| `Behavior` | Suicidal behavior |
| `Attempt` | Suicide attempt |

The model treats these as an ordinal sequence:

```text
Indicator < Ideation < Behavior < Attempt
```

## Factor Taxonomy

The 24 factor categories used by the notebook are:

- Mental health issues
- Physical health/characteristic
- Substance use
- Hopelessness
- Emotion dysregulation
- Low self-esteem
- Poor school performance
- Low socio-economic status
- Interpersonal violence
- Prior self-harm or suicidal thought/attempt
- Poor social support
- Interpersonal difficulty
- Dysfunctional family
- Exposure to others' suicide
- Stressful life event
- Traumatic experience
- Cognitive deficits
- Suicide means (with access)
- Sexual orientation related issues
- Social support
- Coping strategy
- Psychological capital
- Sense of responsibility
- Meaning in life

The last five categories are treated as protective factors in the notebook's LLM annotation prompt.

## Dataset

The notebook expects two Excel files:

```text
train.xlsx
leaderboard.xlsx
```

The training data is expected to contain fields such as:

```text
post
anon_user_id
row_id
suicide risk
factors
evidence for suicide risk level
```

The leaderboard/test file is expected to contain the post information and identifiers needed for prediction.

The notebook cleans HTML entities, normalizes risk labels, removes duplicate factor tags, and ignores placeholder evidence values such as `none` and `n/a`.

## Installation

The notebook installs the main dependencies with:

```bash
pip install -q transformers scikit-learn openpyxl accelerate
```

The implementation also uses packages including:

```text
PyTorch
NumPy
Pandas
Hugging Face Transformers
Hugging Face Hub
scikit-learn
openpyxl
```

## Running the Notebook

The notebook was designed primarily for a **Google Colab GPU runtime**.

### 1. Enable a GPU

A T4 GPU is used by the notebook configuration.

In Colab:

```text
Runtime → Change runtime type → T4 GPU
```

### 2. Upload the data

Upload:

```text
train.xlsx
leaderboard.xlsx
```

to the Colab runtime, or change the configured paths to point to your mounted storage.

### 3. Hugging Face authentication

The notebook includes:

```python
from huggingface_hub import login
login()
```

The configured backbone is:

```python
MODEL_NAME = "mental/mental-roberta-base"
```

If the backbone cannot be loaded, the notebook falls back to:

```python
MODEL_NAME_FALLBACK = "roberta-base"
```

The optional LLM signal uses:

```python
microsoft/Phi-3.5-mini-instruct
```

No API key is required for that model in the notebook's current implementation.

### 4. Run the notebook

After configuring the paths and model settings:

```text
Runtime → Run all
```

The notebook performs:

1. Data loading and cleaning
2. Tokenization and evidence alignment
3. Zero-shot LLM factor classification (optional)
4. Grouped stratified cross-validation
5. Transformer training
6. TF-IDF + Logistic Regression training
7. Out-of-fold prediction collection
8. Threshold and ensemble calibration
9. Leaderboard inference
10. CSV submission generation
11. Error analysis

## Main Configuration

Important settings include:

```python
MODEL_NAME = "mental/mental-roberta-base"
MAX_LENGTH = 512
N_FOLDS = 8
EPOCHS = 10
BATCH_SIZE = 8

FREEZE_BOTTOM_FRACTION = 0.1
FREEZE_EMBEDDINGS = True

BACKBONE_LR = 2e-5
HEAD_LR = 1e-4

SEED = 42

USE_EMA = False
USE_FGM = False

RARE_FACTOR_THRESHOLD = 50
OVERSAMPLE_FACTOR = 3

MIN_SUPPORT_FOR_THRESHOLD_TUNING = 15
FACTOR_THRESHOLD_FBETA = 0.7
FACTOR_OVERPREDICT_RATIO = 2.0

TRAIN_PATH = "train.xlsx"
LEADERBOARD_PATH = "leaderboard.xlsx"
TEAM_NAME = "4aumArivu"
```

## Training Strategy

### Grouped Cross-Validation

The notebook uses:

```python
StratifiedGroupKFold
```

with `anon_user_id` as the grouping variable.

This prevents posts from the same user from being distributed across training and validation folds.

### Rare-Factor Handling

Rows containing rare factor categories are oversampled during training.

The default configuration is:

```python
RARE_FACTOR_THRESHOLD = 50
OVERSAMPLE_FACTOR = 3
```

Factor loss additionally uses alpha-weighted focal loss to address class imbalance.

### Risk Prediction

Risk uses two heads:

- Standard 4-class classifier
- CORAL ordinal classifier

Their probability distributions are blended before selecting the final risk class.

### Evidence Extraction

Evidence is predicted at the token level.

The tokenizer's character offsets are used to reconstruct the original text spans. Predicted spans can be merged when separated by a small token gap.

The evidence decoding parameters are tuned against the project's Phrase-F1 metric using pooled out-of-fold predictions.

### Factor Prediction Ensemble

The factor probabilities can combine:

1. Transformer model
2. Classical TF-IDF + Logistic Regression
3. Optional zero-shot Phi-3.5-mini-instruct predictions

The blend weight is selected using pooled out-of-fold predictions.

## Evaluation

The notebook calculates three task-specific metrics:

### Risk

Weighted F1:

```text
Weighted F1
```

### Evidence

Phrase-level F1 using one-to-one matching between predicted and gold evidence spans.

The implementation also enforces the project's compatibility rules, including the 3× token-length cap.

### Factors

Macro F1 across the 24 factor categories.

### Composite Score

The notebook combines the three metrics as:

```text
Composite =
    0.4 × Risk F1
  + 0.3 × Evidence F1
  + 0.3 × Factors F1
```

## Output

The final prediction file is generated using:

```python
out_path = f"{TEAM_NAME}.csv"
```

With the default configuration:

```text
4aumArivu.csv
```

The output contains:

```text
row_id
risk_level
evidence
factors
```

Example structure:

```csv
row_id,risk_level,evidence,factors
...,Ideation,"example evidence span","['hopelessness', 'poor social support']"
```

## Error Analysis

The notebook includes:

- Risk-level confusion matrix
- Classification report
- Per-category factor precision
- Per-category factor recall
- Per-category factor F1
- True prevalence
- Predicted rate
- Identification of weak factor categories

This is based on out-of-fold predictions so that each training row is evaluated by a model that did not train on that row.

## Project File

The main notebook is:

```text
suicide_risk_transformer__26_(1).ipynb
```

Recommended repository structure:

```text
suicide-risk-prediction-/
│
├── suicide_risk_transformer__26_(1).ipynb
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── train.xlsx
│   └── leaderboard.xlsx
│
└── outputs/
    └── 4aumArivu.csv
```

> Do not commit private datasets, API tokens, credentials, or other sensitive information to a public repository.

## Limitations

- The notebook truncates inputs at 512 tokens.
- Very long posts may therefore have evidence beyond the truncation point that cannot be recovered.
- Rare factor categories have limited training examples.
- Cross-validation performance is not equivalent to performance on an independent real-world population.
- The optional LLM factor classifier can take significant time to run.
- GPU memory and runtime requirements can be substantial.
- Mental-health and suicide-related language is highly contextual; automated predictions can be incorrect.
- The model should not be used as the sole basis for clinical, emergency, moderation, or other high-stakes decisions.

## Responsible Use

This repository involves sensitive mental-health and suicide-related text.

When using or extending the project:

- Protect the privacy of source data.
- Do not expose personally identifying information.
- Do not publish private posts without appropriate authorization.
- Treat predictions as model outputs rather than confirmed assessments.
- Include qualified human review for any high-stakes application.
- Evaluate bias and performance across relevant populations before deployment.

## Acknowledgements

The transformer pipeline uses the Hugging Face ecosystem and the `mental/mental-roberta-base` model as its primary encoder.

The optional zero-shot factor signal uses:

```text
microsoft/Phi-3.5-mini-instruct
```

## License

Add the project's intended license here before publishing the repository, for example:

```text
MIT License
```

Make sure the selected license is compatible with all datasets, pretrained models, and other third-party components used by the project.
