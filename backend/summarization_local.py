from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

# Paths for fine-tuned model
FINE_TUNED_MODEL_DIR = "models/fine_tuned_summarizer"

# Check if fine-tuned model exists, else use pretrained
if os.path.exists(FINE_TUNED_MODEL_DIR):
    model = AutoModelForSeq2SeqLM.from_pretrained(FINE_TUNED_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
else:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)

def summarize_text(text: str, max_length=150, min_length=50):
    result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return result[0]['summary_text']
