import gradio as gr
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)

def predict_sentiment(text):
    if not text.strip():
        return " Please enter some text."

    result = classifier(text)[0]
    label = result['label']
    score = round(result['score'], 4)

    emoji = "😃" if label == "POSITIVE" else "😠"
    verdict = f"{emoji} {label} ({score * 100:.1f}% confidence)"

    return verdict

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Paste your Ex's last message,Texts, tweet, or customer rant here...",
        label="Enter Text"
    ),
    outputs=gr.Text(label=" Anguzu's AI Sentiment Analysis"),
    title="Anguzu Sentiment AI",
    description="""
Built by Anguzu,I believer emotions matter, and data doesn't lie.  
This AI uses transformer-based deep learning to classify text as either positive or negative.  
Try me on feedback, tweets, product reviews, or even your ex’s last message .
""",
    theme="soft",
    allow_flagging="never"
)

demo.launch()

