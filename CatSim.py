# Use OpenAI embedding model (via langchain) and categories from categorie.txt to rate each category for each URL.
# MAXIMUM SPEED: vectorized numpy ops, largest safe batches, thread pools, and tqdm everywhere.
# For Mac: %pip3 install -qU langchain-openai tqdm

import os
import dotenv
import json
import numpy as np
from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
import concurrent.futures

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

with open('Results/ResultsData.json', 'r') as f:
    data = json.load(f)

with open('Input/categories.txt', 'r') as f:
    categories = [cat.strip() for cat in f.read().split(';') if cat.strip()]

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
category_embeddings = embeddings.embed_documents(categories)
category_embeddings = np.stack([np.array(emb, dtype=np.float32) for emb in category_embeddings])
category_embeddings /= np.linalg.norm(category_embeddings, axis=1, keepdims=True) + 1e-10

field_names = ["url", "title", "desc", "content"]
weights = np.array([0.1, 0.15, 0.35, 0.4], dtype=np.float32)

# Prepare all field texts for all entries for batch embedding
all_texts = []
entry_field_indices = []
for entry_idx, entry in enumerate(data):
    for field in field_names:
        text = entry.get(field, "")
        all_texts.append(text.strip() if isinstance(text, str) and len(text.strip()) >= 5 else "")
        entry_field_indices.append((entry_idx, field))

def count_tokens(text):
    return max(1, len(text) // 4)

MAX_TOKENS_PER_BATCH = 10_000  # Use as much as possible for max speed

def batch_texts_by_token_limit(texts, max_tokens):
    batches, current_batch, current_tokens = [], [], 0
    for text in texts:
        tokens = count_tokens(text)
        if current_tokens + tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch, current_tokens = [], 0
        current_batch.append(text)
        current_tokens += tokens
    if current_batch:
        batches.append(current_batch)
    return batches

batches = batch_texts_by_token_limit(all_texts, MAX_TOKENS_PER_BATCH)

# Embed all texts in largest possible batches
all_embeddings = []
for batch in tqdm(batches, desc="Embedding batches", mininterval=0.1):
    all_embeddings.extend(embeddings.embed_documents(batch))
all_embeddings = np.stack([np.array(emb, dtype=np.float32) for emb in all_embeddings])

embedding_dim = category_embeddings.shape[1]
num_entries = len(data)
num_fields = len(field_names)

# Map embeddings to entries/fields, normalize, and mask empty fields
field_embeddings_per_entry = np.zeros((num_entries, num_fields, embedding_dim), dtype=np.float32)
field_nonempty_mask = np.zeros((num_entries, num_fields), dtype=bool)
for idx, (entry_idx, field) in enumerate(entry_field_indices):
    emb = all_embeddings[idx]
    if all_texts[idx].strip():
        norm_emb = emb / (np.linalg.norm(emb) + 1e-10)
        field_embeddings_per_entry[entry_idx, field_names.index(field)] = norm_emb
        field_nonempty_mask[entry_idx, field_names.index(field)] = True

# Vectorized scoring function for a single entry
def score_entry(entry_idx):
    field_embs = field_embeddings_per_entry[entry_idx]  # (num_fields, emb_dim)
    mask = field_nonempty_mask[entry_idx]               # (num_fields,)
    if not np.any(mask):
        return {
            "url": data[entry_idx].get("url", ""),
            "top5_category_ratings": {}
        }
    # (num_fields, emb_dim) dot (emb_dim, num_categories) -> (num_fields, num_categories)
    sims = np.dot(field_embs, category_embeddings.T)    # (num_fields, num_categories)
    sims[~mask, :] = 0.0
    used_weights = weights * mask
    total_weight = used_weights.sum()
    if total_weight < 1e-8:
        return {
            "url": data[entry_idx].get("url", ""),
            "top5_category_ratings": {}
        }
    # Weighted sum over fields
    weighted_sims = (used_weights[:, None] * sims).sum(axis=0) / total_weight  # (num_categories,)
    # Top N categories
    topN_idx = np.argsort(weighted_sims)[::-1][:10]
    topN = [(categories[i], float(weighted_sims[i])) for i in topN_idx if weighted_sims[i] >= 0.25]
    return {
        "url": data[entry_idx].get("url", ""),
        "top5_category_ratings": dict(topN)
    }

# Parallelize scoring with ThreadPoolExecutor
ratings = []
with concurrent.futures.ThreadPoolExecutor() as executor, tqdm(total=num_entries, desc="Rating URLs (max speed)", mininterval=0.1) as pbar:
    for result in executor.map(score_entry, range(num_entries), chunksize=32):
        ratings.append(result)
        pbar.update(1)

with open('Results/ratings.json', 'w') as f:
    json.dump(ratings, f, indent=2)
