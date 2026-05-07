import os
from src.retriever import retrieve
from src.storage import STORAGE_DIR
import json


def load_persona():
    try:
        with open(os.path.join(STORAGE_DIR, "persona.json")) as f:
            persona = json.load(f)
    except FileNotFoundError:
        persona = {
            "global": {
                "habits": [],
                "personality": [],
                "communication_style": [],
                "facts": []
            },
            "user1": {
                "habits": [],
                "personality": [],
                "communication_style": [],
                "facts": []
            },
            "user2": {
                "habits": [],
                "personality": [],
                "communication_style": [],
                "facts": []
            }
        }

    if "global" not in persona and "user1" not in persona and "user2" not in persona:
        persona = {
            "global": persona,
            "user1": persona,
            "user2": persona
        }

    return persona


def user_persona_key(query_lower):
    if "user 1" in query_lower or "user1" in query_lower:
        return "user1"
    if "user 2" in query_lower or "user2" in query_lower:
        return "user2"
    return "global"


def answer(query):
    persona_data = load_persona()
    results = retrieve(query)

    topic_context = "\n".join([t["summary"] for t in results.get("topics", [])])
    checkpoint_context = "\n".join([c["summary"] for c in results.get("checkpoints", [])])
    query_lower = query.lower()
    persona = persona_data.get(user_persona_key(query_lower), persona_data.get("global", {}))

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