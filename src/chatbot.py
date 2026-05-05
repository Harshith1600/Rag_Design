import os
from src.retriever import retrieve
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")


def load_persona():
    try:
        with open(os.path.join(STORAGE_DIR, "persona.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "habits": [],
            "personality": [],
            "communication_style": [],
            "facts": []
        }


def answer(query):
    persona = load_persona()
    results = retrieve(query)

    topic_context = "\n".join([t["summary"] for t in results.get("topics", [])])
    checkpoint_context = "\n".join([c["summary"] for c in results.get("checkpoints", [])])
    query_lower = query.lower()

    if any(k in query_lower for k in ["what kind of person", "type of person", "describe the person", "how are they", "personality"]):
        return (
            f"Personality: {persona.get('personality', [])}\n"
            f"Habits: {persona.get('habits', [])}\n"
            f"Communication style: {persona.get('communication_style', [])}\n"
            f"Facts: {persona.get('facts', [])}"
        )

    if "habit" in query_lower:
        return f"Habits: {persona.get('habits', [])}"

    if any(k in query_lower for k in ["how do they talk", "communication", "talks", "speech"]):
        return f"Communication style: {persona.get('communication_style', [])}"

    if "person" in query_lower and not any(k in query_lower for k in ["habit", "communication", "persona", "personality"]):
        return (
            f"Persona summary:\nPersonality: {persona.get('personality', [])}\n"
            f"Habits: {persona.get('habits', [])}\n"
            f"Communication style: {persona.get('communication_style', [])}\n"
            f"Facts: {persona.get('facts', [])}"
        )

    return (
        f"Relevant Topic Summaries:\n{topic_context}\n\n"
        f"Relevant 100-message summaries:\n{checkpoint_context}\n\n"
        "Answer:\nBased on the retrieved context, this is the best possible answer."
    )