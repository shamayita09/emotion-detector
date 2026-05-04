# Emotion detector — DistilBERT fine-tuned on GoEmotions

A text emotion classification model fine-tuned on Google's GoEmotions dataset. Given any sentence, it predicts the most likely emotion from 28 categories including joy, sadness, anger, fear, surprise, and more.

**Live demo:** [huggingface.co/spaces/Shamayita09/emotion-detector](https://huggingface.co/spaces/Shamayita09/emotion-detector)  
**Model:** [huggingface.co/Shamayita09/emotion-distilbert](https://huggingface.co/Shamayita09/emotion-distilbert)

---

## Example predictions

| Input | Top emotion | Confidence |
|---|---|---|
| "I just got accepted into my dream university!" | approval | 49.1% |
| "I can't believe they cancelled my favourite show." | surprise | 74.4% |
| "I'm so nervous about my exam tomorrow." | nervousness | 53.1% |

---

## Model details

| | |
|---|---|
| Base model | `distilbert-base-uncased` |
| Dataset | GoEmotions (simplified), 43,410 training examples |
| Labels | 28 emotion categories |
| Training epochs | 3 |
| Validation accuracy | 57.6% |
| Validation F1 | 0.569 (weighted) |
| Hardware | Google Colab T4 GPU |
| Training time | ~25 minutes |

---

## Industry use cases

This type of model powers real products across multiple industries:

- **Customer support** — auto-flag frustrated tickets for immediate escalation
- **Social media monitoring** — track brand sentiment beyond positive/negative
- **Mental health apps** — detect distress signals in user messages
- **HR platforms** — analyse employee survey responses for burnout signals
- **Education** — detect confusion or disengagement in student responses

---

## How to run locally

```bash
pip install transformers gradio datasets torch
```

```python
from transformers import pipeline
from datasets import load_dataset

dataset = load_dataset("go_emotions", "simplified")
label_names = dataset["train"].features["labels"].feature.names

classifier = pipeline(
    "text-classification",
    model="Shamayita09/emotion-distilbert",
    top_k=None
)

def predict(text):
    results = classifier(text)[0]
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    for r in results[:5]:
        label = label_names[int(r["label"].split("_")[-1])]
        print(f"{label:20} {round(r['score']*100, 1)}%")

predict("I'm so happy today!")
```

---

## Project structure

```
emotion-distilbert/
├── app.py              # Gradio demo
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## What I learned

- Fine-tuning a pre-trained transformer (DistilBERT) for a downstream classification task
- Working with the HuggingFace `transformers` and `datasets` libraries
- Handling class imbalance in real-world datasets
- Deploying an ML model as a live web app using Gradio and Hugging Face Spaces

---

## Built with

- [HuggingFace Transformers](https://huggingface.co/transformers)
- [GoEmotions dataset](https://huggingface.co/datasets/go_emotions) by Google Research
- [Gradio](https://gradio.app)
- [PyTorch](https://pytorch.org)
