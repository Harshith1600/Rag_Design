import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress HuggingFace Hub authentication warning
warnings.filterwarnings("ignore", message=".*You are sending unauthenticated requests.*")

from sentence_transformers import SentenceTransformer

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading embedding model: {e}")
    model = None


def embed(texts):
    if model is None:
        raise RuntimeError("Embedding model is not available")
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
