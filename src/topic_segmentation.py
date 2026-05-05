import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.embeddings import embed

THRESHOLD = 0.60
WINDOW = 5
MIN_TOPIC_CHUNKS = 3
CHUNK_SIZE = 10

def keyword_overlap(a, b):
    set1 = set(a.lower().split())
    set2 = set(b.lower().split())
    return len(set1 & set2) / (len(set1 | set2) + 1e-5)

def is_question(text):
    return "?" in text or text.lower().startswith(("what", "why", "how", "explain"))

def batch_embed_texts(texts, batch_size=512):
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start+batch_size]
        embeddings.append(embed(batch))
    return np.vstack(embeddings)


def segment_topics(messages):
    topics = []
    current = []
    current_chunk_count = 0
    topic_id = 0

    chunk_texts = [
        " ".join([m["text"] for m in messages[i:i+CHUNK_SIZE]])
        for i in range(0, len(messages), CHUNK_SIZE)
    ]
    message_chunks = [messages[i:i+CHUNK_SIZE] for i in range(0, len(messages), CHUNK_SIZE)]

    print(f"Embedding {len(chunk_texts)} chunks for topic segmentation...")
    chunk_embeddings = batch_embed_texts(chunk_texts)

    for idx, chunk in enumerate(message_chunks):
        emb = chunk_embeddings[idx]

        if current_chunk_count >= WINDOW:
            prev_embs = chunk_embeddings[idx-WINDOW:idx]
            avg_emb = np.mean(prev_embs, axis=0)
            sim = cosine_similarity([emb], [avg_emb])[0][0]

            prev_text = current[-1]["text"]
            keyword_sim = keyword_overlap(chunk[-1]["text"], prev_text)
            intent_shift = is_question(chunk[-1]["text"]) != is_question(prev_text)

            if (sim < THRESHOLD and keyword_sim < 0.2) or intent_shift:
                if current_chunk_count >= MIN_TOPIC_CHUNKS:
                    topics.append({
                        "topic_id": topic_id,
                        "messages": current
                    })
                    topic_id += 1
                    current = []
                    current_chunk_count = 0

        current.extend(chunk)
        current_chunk_count += 1

    if current:
        topics.append({
            "topic_id": topic_id,
            "messages": current
        })

    return topics