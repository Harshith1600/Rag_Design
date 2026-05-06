import os
import json
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
MAX_FACTS = 8

HABIT_PATTERNS = {
    "coffee drinker": ["coffee", "latte", "espresso"],
    "early morning routine": ["morning", "early morning", "wake up"],
    "fitness oriented": ["run", "hike", "yoga", "workout", "gym"],
    "home cooking": ["cook", "cooking", "kitchen", "recipe"],
    "traveler": ["moving to", "moving", "travel", "airport", "fly"]
}

PERSONALITY_PATTERNS = {
    "humorous": ["lol", "haha", "😂", "funny", "lmao", "rofl", "joke", "hilarious"],
    "emotional": ["sad", "upset", "angry", "happy", "excited", "stress", "stressed"],
    "friendly": ["hi", "hello", "hey", "thanks", "thank you"]
}

COMMUNICATION_PATTERNS = {
    "short messages": lambda text: len(text.split()) <= 5,
    "enthusiastic": lambda text: any(ch in text for ch in ["!", "😊", "😂", "😍"]),
    "question-oriented": lambda text: text.strip().endswith("?")
}

FACT_PATTERNS = ["i am", "i'm", "my name", "i have", "i've", "i was", "i will"]


def normalize_fact(text):
    return " ".join(text.strip().split())


def compute_persona(messages):
    habits = Counter()
    personality = Counter()
    communication_style = Counter()
    facts = []

    for m in messages:
        text = m["text"]
        lower = text.lower()

        for label, patterns in HABIT_PATTERNS.items():
            if any(pattern in lower for pattern in patterns):
                habits[label] += 1

        for label, patterns in PERSONALITY_PATTERNS.items():
            if any(pattern in lower for pattern in patterns):
                personality[label] += 1

        for label, matcher in COMMUNICATION_PATTERNS.items():
            if matcher(text):
                communication_style[label] += 1

        if len(facts) < MAX_FACTS and any(p in lower for p in FACT_PATTERNS):
            facts.append(normalize_fact(text))

    return {
        "habits": [habit for habit, _ in habits.most_common(5)],
        "personality": [trait for trait, _ in personality.most_common(5)],
        "communication_style": [style for style, _ in communication_style.most_common(5)],
        "facts": list(dict.fromkeys(facts))
    }


def extract_persona(messages):
    os.makedirs(STORAGE_DIR, exist_ok=True)

    persona_data = {
        "global": compute_persona(messages),
        "user1": compute_persona([m for m in messages if m.get("user") == "user1"]),
        "user2": compute_persona([m for m in messages if m.get("user") == "user2"])
    }

    with open(os.path.join(STORAGE_DIR, "persona.json"), "w") as f:
        json.dump(persona_data, f, indent=2)

    return persona_data