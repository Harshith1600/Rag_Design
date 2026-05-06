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
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return text[:200].strip()
    return " ".join(sentences[:max_sent])

def extract_keywords(messages, top_k=3):
    words = normalize(" ".join([m["text"] for m in messages])).split()
    words = [w for w in words if w and w not in STOP_WORDS]
    common = Counter(words).most_common(top_k)
    return [w for w, _ in common]

def label_topic(messages):
    keywords = extract_keywords(messages)
    return " / ".join(keywords)