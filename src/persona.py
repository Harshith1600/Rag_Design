import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")


def extract_persona(messages):
    os.makedirs(STORAGE_DIR, exist_ok=True)

    persona = {
        "habits": [],
        "personality": [],
        "communication_style": [],
        "facts": []
    }

    for m in messages:
        text = m["text"]
        lower = text.lower()

        if any(x in lower for x in ["sleep", "late night", "early morning", "morning", "coffee", "yoga", "run", "hike", "cook", "cooking"]):
            persona["habits"].append(text)

        if any(x in lower for x in ["lol", "haha", "😂", "funny", "lmao", "rofl", "joke", "hilarious"]):
            persona["personality"].append("humorous")

        if any(x in lower for x in ["sad", "upset", "angry", "happy", "excited", "stress", "stressed"]):
            persona["personality"].append("emotional")

        if any(x in lower for x in ["i am", "i'm", "my name", "i have", "i've", "i was", "i will"]):
            persona["facts"].append(text)

        if len(text.split()) <= 5:
            persona["communication_style"].append("short messages")
        if any(ch in text for ch in ["!", "😊", "😂", "😍"]):
            persona["communication_style"].append("enthusiastic")
        if text.endswith("?"):
            persona["communication_style"].append("question-oriented")

    for k in persona:
        persona[k] = list(dict.fromkeys(persona[k]))

    with open(os.path.join(STORAGE_DIR, "persona.json"), "w") as f:
        json.dump(persona, f, indent=2)

    return persona