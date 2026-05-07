# Rag_Design

## Overview

`Rag_Design` is a retrieval-augmented conversational analysis project. It detects topic changes in a conversation, builds retrieval indexes over summaries, and generates persona data from message history.

---

## How topic changes are detected

Topic segmentation is implemented in `src/topic_segmentation.py`.

Key steps:

1. Messages are grouped into fixed-size chunks of `CHUNK_SIZE = 10` messages.
2. Each chunk is embedded with `src.embeddings.embed(...)`.
3. A sliding window of the last `WINDOW = 5` chunks is used to compute an average embedding.
4. The current chunk is compared to that average using cosine similarity.
5. A new topic boundary is created when:
   - semantic similarity drops below `THRESHOLD = 0.60`, and
   - keyword overlap is low, or
   - the conversational intent changes (question vs non-question).

Snippet:

```python
THRESHOLD = 0.60
WINDOW = 5
MIN_TOPIC_CHUNKS = 3
CHUNK_SIZE = 10

for idx, chunk in enumerate(message_chunks):
    emb = chunk_embeddings[idx]

    if current_chunk_count >= WINDOW:
        prev_embs = chunk_embeddings[idx-WINDOW:idx]
        avg_emb = np.mean(prev_embs, axis=0)
        sim = cosine_similarity([emb], [avg_emb])[0][0]

        prev_text = current[-1]["text"]
        keyword_sim = keyword_overlap(chunk[-1]["text"], prev_text)
        intent_shift = is_question(chunk[-1]["text"]) != is_question(prev_text)

        if (sim < THRESHOLD and keyword_sim < 0.2) or intent_shift:
            if current_chunk_count >= MIN_TOPIC_CHUNKS:
                topics.append({"topic_id": topic_id, "messages": current})
                topic_id += 1
                current = []
                current_chunk_count = 0
```

This means topic changes are based on both semantic drift and conversational intent.

---

## How retrieval works

Retrieval is implemented in `src/retriever.py`.

How it works:

- `load()` reads the FAISS index and metadata from `storage/`.
- The query is embedded using `src.embeddings.embed(query)`.
- FAISS returns the top `k` nearest chunks.
- Those chunk ids are mapped back to the saved topic summaries and checkpoint summaries.

Snippet:

```python
q_emb = np.array([embed(query)]).astype("float32")
D, I = index.search(q_emb, k)
for idx in I[0]:
    if idx < 0:
        continue
    topic_id = id_map.get(str(int(idx)))
    topic = next((t for t in topics if t["topic_id"] == topic_id), None)
    if topic:
        results.append(topic)
```

Supported retrieval results:

- `topics` from `topic_summaries.json`
- `checkpoints` from `checkpoints_100.json`

This is the part that gives the chatbot relevant context for an answer.

---

## How persona is built

Persona building is handled in `src/persona.py`.

The process:

1. Conversation messages are parsed in `src/preprocess.py` to extract `user1` and `user2` labels.
2. `extract_persona(messages)` computes three persona records:
   - `global`
   - `user1`
   - `user2`
3. Each persona record is built from message text using pattern matching.

Pattern categories:

- Habits (`HABIT_PATTERNS`)
- Personality traits (`PERSONALITY_PATTERNS`)
- Communication style (`COMMUNICATION_PATTERNS`)
- Facts detected from phrases like `"i am"`, `"i'm"`, `"my name"`

Snippet:

```python
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
```

Finally, `extract_persona(...)` saves the output to `storage/persona.json`.

---

## How to run the pipeline

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Build the pipeline and generate artifacts:

```bash
python src/build_pipeline.py
```

3. Start the Flask app:

```bash
python app.py
```

4. Use the UI or POST to `/chat` with JSON `{ "query": "..." }`.

---

## Render persistent disk

To keep the generated `storage/` artifacts across restarts, use a Render persistent disk.

1. In your Render service settings, add a persistent disk and mount it at `storage/` inside the service.
2. In `render.yaml`, the service includes an environment variable:

```yaml
      - key: STORAGE_DIR
        value: storage
```

3. The app and pipeline now use `STORAGE_DIR` if it is set.
4. This keeps files like:
   - `storage/persona.json`
   - `storage/faiss_index.bin`
   - `storage/id_map.json`
   - `storage/topic_summaries.json`
   - `storage/checkpoints_100.json`
   - `storage/checkpoints_100_index.bin`
   - `storage/checkpoints_100_id_map.json`

If the disk is mounted somewhere else, update `STORAGE_DIR` in Render environment variables to match the mount path.

---

## Notes

- If the query mentions `user1` or `user2`, the chatbot uses that user-specific persona data.
- If no specific user is mentioned, it falls back to the global persona summary.
- Topic changes are detected from both semantic similarity and conversation structure, not only raw keywords.

