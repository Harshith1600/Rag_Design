import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import faiss
import numpy as np
from src.embeddings import embed
from src.summarizer import summarize, label_topic
from src.storage import STORAGE_DIR


def build_all(topics):
    os.makedirs(STORAGE_DIR, exist_ok=True)

    topic_data = []
    vectors = []
    id_map = {}

    for t in topics:
        msgs = t["messages"]

        summary = summarize(msgs)
        label = label_topic(msgs)

        emb = np.asarray(embed(summary), dtype="float32")

        topic_data.append({
            "topic_id": t["topic_id"],
            "start": msgs[0]["id"],
            "end": msgs[-1]["id"],
            "label": label,
            "summary": summary
        })

        vectors.append(emb)
        id_map[str(len(vectors) - 1)] = t["topic_id"]

    with open(os.path.join(STORAGE_DIR, "topic_summaries.json"), "w") as f:
        json.dump(topic_data, f, indent=2)

    if vectors:
        dim = vectors[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.vstack(vectors))
        faiss.write_index(index, os.path.join(STORAGE_DIR, "faiss_index.bin"))

        with open(os.path.join(STORAGE_DIR, "id_map.json"), "w") as f:
            json.dump(id_map, f, indent=2)

