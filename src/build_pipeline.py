import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import load_messages
from src.topic_segmentation import segment_topics
from src.topic_builder import build_all
from src.build_100_checkpoints import build_100_checkpoints
from src.persona import extract_persona

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "conversations.csv")


def build_all_data(data_path=DATA_PATH):
    messages = load_messages(data_path)
    print(f"Loaded {len(messages)} messages from dataset")

    topics = segment_topics(messages)
    print(f"Detected {len(topics)} topic segments")

    build_all(topics)
    print("Saved topic summaries and FAISS index")

    build_100_checkpoints(messages)
    print("Saved 100-message checkpoint summaries and index")

    extract_persona(messages)
    print("Saved persona json")

    return messages, topics


if __name__ == "__main__":
    try:
        build_all_data()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise