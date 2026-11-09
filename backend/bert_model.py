from sentence_transformers import SentenceTransformer, util

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Store embeddings cache
corpus_embeddings = None
corpus = []

def prepare_bert(df):
    global corpus_embeddings, corpus

    corpus = df["text"].tolist()
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)


def get_bert_response(user_input, df):
    global corpus_embeddings, corpus

    if corpus_embeddings is None:
        return None

    query_embedding = model.encode(user_input, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()

    # Threshold (important)
    if best_score < 0.4:
        return None

    row = df.iloc[best_idx]

    reply = f"📌 {row['Description']}"

    if row.get("Extra Info"):
        reply += f"<br><br><b>Extra Info:</b> {row['Extra Info']}"

    return reply