import os
import json
import faiss
import numpy as np
from src.embeddings import embed

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")


def load():
    index = faiss.read_index(os.path.join(STORAGE_DIR, "faiss_index.bin"))

    with open(os.path.join(STORAGE_DIR, "id_map.json")) as f:
        id_map = json.load(f)

    with open(os.path.join(STORAGE_DIR, "topic_summaries.json")) as f:
        topics = json.load(f)

    checkpoints = []
    cp_index = None
    cp_id_map = {}
    checkpoints_path = os.path.join(STORAGE_DIR, "checkpoints_100.json")
    cp_index_path = os.path.join(STORAGE_DIR, "checkpoints_100_index.bin")
    cp_id_map_path = os.path.join(STORAGE_DIR, "checkpoints_100_id_map.json")

    if os.path.exists(checkpoints_path):
        with open(checkpoints_path) as f:
            checkpoints = json.load(f)

    if os.path.exists(cp_index_path) and os.path.exists(cp_id_map_path):
        cp_index = faiss.read_index(cp_index_path)
        with open(cp_id_map_path) as f:
            cp_id_map = json.load(f)

    return index, id_map, topics, cp_index, cp_id_map, checkpoints


def retrieve(query, k=3):
    try:
        index, id_map, topics, cp_index, cp_id_map, checkpoints = load()
    except Exception as e:
        print(f"Retriever load error: {e}")
        return {"topics": [], "checkpoints": []}

    q_emb = np.array([embed(query)]).astype("float32")

    results = []
    D, I = index.search(q_emb, k)
    for idx in I[0]:
        if idx < 0:
            continue
        topic_id = id_map.get(str(int(idx)))
        if topic_id is None:
            continue
        topic = next((t for t in topics if t["topic_id"] == topic_id), None)
        if topic:
            results.append(topic)

    checkpoint_results = []
    if cp_index is not None:
        D2, I2 = cp_index.search(q_emb, k)
        for idx in I2[0]:
            if idx < 0:
                continue
            cp_id = cp_id_map.get(str(int(idx)))
            if cp_id is None:
                continue
            checkpoint = next((c for c in checkpoints if c["chunk_id"] == cp_id), None)
            if checkpoint:
                checkpoint_results.append(checkpoint)

    return {"topics": results, "checkpoints": checkpoint_results}
