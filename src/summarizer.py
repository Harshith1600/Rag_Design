import re
from collections import Counter

STOP_WORDS = {
    "the", "and", "is", "in", "to", "of", "a", "for", "on", "it", "that", "this", "with",
    "as", "an", "are", "be", "was", "were", "at", "by", "from", "or", "but", "if", "they"
}


def normalize(text):
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return cleaned


def summarize(messages, max_sent=2):
    text = " ".join([m["text"] for m in messages]).strip()
    sentences = [s.strip() for s in re.split(r'(?<=