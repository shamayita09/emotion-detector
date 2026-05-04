import gradio as gr
from transformers import pipeline
from datasets import load_dataset

dataset = load_dataset("go_emotions", "simplified")
label_names = dataset["train"].features["labels"].feature.names

classifier = pipeline(
    "text-classification",
    model="Shamayita09/emotion-distilbert",
    top_k=None
)

def gradio_predict(text):
    results = classifier(text)[0]
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {
        label_names[int(r["label"].split("_")[-1])]: round(r["score"], 3)
        for r in results
    }

demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(
        placeholder="Type any sentence here...",
        label="Input text"
    ),
    outputs=gr.Label(num_top_classes=5),
    title="Emotion detector",
    description="Detects emotion in text using DistilBERT fine-tuned on GoEmotions",
    examples=[
        ["I just got accepted into my dream university!"],
        ["I can't believe they cancelled my favourite show."],
        ["I'm so nervous about my exam tomorrow."],
        ["I am so grateful for everything I have."],
        ["This is the worst day of my life."],
    ]
)

demo.launch()
