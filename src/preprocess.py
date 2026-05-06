import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_messages(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "conversations.csv")
    df = pd.read_csv(path, header=None, engine="python")

    messages = []
    msg_id = 0

    for _, row in df.iterrows():
        if len(row) == 0 or pd.isna(row.iloc[0]):
            continue
        convo = str(row.iloc[0]).split("\n")

        for msg in convo:
            msg = msg.strip()
            if not msg:
                continue

            speaker = "unknown"
            text = msg
            lowered = msg.lower()
            if lowered.startswith("user 1:"):
                speaker = "user1"
                text = msg[len("user 1:"):].strip()
            elif lowered.startswith("user 2:"):
                speaker = "user2"
                text = msg[len("user 2:"):].strip()

            messages.append({
                "id": msg_id,
                "text": text,
                "user": speaker
            })
            msg_id += 1

    return messages