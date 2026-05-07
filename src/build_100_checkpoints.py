import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import faiss
import numpy as np
from src.embeddings import embed
from src.summarizer import summarize
from src.storage import STORAGE_DIR


def build_100_checkpoints(messages):
    os.makedirs(STORAGE_DIR, exist_ok=True)

    checkpoints = []
    vectors = []
    id_map = {}
    chunk_id = 0

    for i in range(0, len(messages), 100):
        chunk = messages[i:i+100]
        summary = summarize(chunk, max_sent=2)

        checkpoints.append({
            "chunk_id": chunk_id,
            "range": f"{i}-{i+len(chunk)}",
            "summary": summary
        })

        embedding = np.asarray(embed(summary), dtype="float32")
        vectors.append(embedding)
        id_map[str(chunk_id)] = chunk_id
        chunk_id += 1

    with open(os.path.join(STORAGE_DIR, "checkpoints_100.json"), "w") as f:
        json.dump(checkpoints, f, indent=2)

    if vectors:
        dim = vectors[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.vstack(vectors))
        faiss.write_index(index, os.path.join(STORAGE_DIR, "checkpoints_100_index.bin"))

        with open(os.path.join(STORAGE_DIR, "checkpoints_100_id_map.json"), "w") as f:
            json.dump(id_map, f, indent=2)
