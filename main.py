from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import joblib
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
model={}
docs_index={}
doc_ids={}
docs={}
doc_embeddings={}

model['quora'] = SentenceTransformer(f'quora_mpnet_v2_tuned_v3')
docs_index['quora'] = faiss.read_index(f'quora_faiss_index_v3.index')
doc_ids['quora'] = joblib.load(f"all_quora_doc_ids_v3.joblib")
docs['quora'] = joblib.load(f"all_quora_doc_texts.joblib")
doc_embeddings['quora'] = joblib.load(f"all_quora_doc_embeddings_v3.joblib") 



model['antique'] = SentenceTransformer(f'antique/antique_mpnet_v2_tuned_v3')
docs_index['antique'] = faiss.read_index(f'antique/antique_faiss_index_v3.index')
doc_ids['antique'] = joblib.load(f"antique/all_antique_doc_ids_v3.joblib")
docs['antique'] = joblib.load(f"antique/all_antique_doc_texts.joblib")
doc_embeddings['antique'] = joblib.load(f"antique/all_antique_doc_embeddings_v3.joblib") 



from sklearn.metrics.pairwise import cosine_similarity

def search_without_faiss(dataset, query, top_k=5):
    # Encode query
    # model = SentenceTransformer(f'{dataset}_mpnet_v2_tuned_v3')
    query_embedding = model[dataset].encode([query])
    query_embedding_np = np.array(query_embedding).astype('float32')

    # Load docs adn docs ids
    doc_embeddings[dataset]= np.array(doc_embeddings[dataset]).astype('float32')
    
    # Compute cosine similarities
    similarities = cosine_similarity(query_embedding, doc_embeddings[dataset])[0]  # shape: (num_docs,)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Collect top-k results
    results = []
    for idx in top_indices:
        results.append({
            "doc_id": doc_ids[dataset][idx],
            "text": docs[dataset][idx],
            "score": float(similarities[idx])
        })
    return results


# search function
def search_faiss(dataset, query, top_k=5):
    # Encode query
    query_embedding = model[dataset].encode([query])
    query_embedding_np = np.array(query_embedding).astype('float32')

    # Load Faiss, docs, docs_ids

    # Search in FAISS index
    D, I = docs_index[dataset].search(query_embedding_np, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        results.append({
            "doc_id": doc_ids[dataset][idx],
            "text": docs[dataset][idx],
            "score": 1 - float(score)
        })

    # Sort by score descending (higher score first)
    results.sort(key=lambda x: x["score"], reverse=True)

    return results

@app.post("/search")
def search(query: str = Form(...), dataset: str = Form(...)):
    results = search_faiss(dataset,query, top_k=10)
    return {"results": results}



