import numpy as np
from utils.store_in_chroma import add_embedding_to_chroma
import chromadb
from chromadb.config import Settings

# Connect to ChromaDB
client = chromadb.Client(Settings(
    persist_directory="db/chroma",
    anonymized_telemetry=False
))

collection = client.get_or_create_collection(name="face_embeddings")

# Test data
test_user_id = "علي الكرار"
test_embedding = np.random.rand(128)  # Replace with your actual face embedding if you want

# Add embedding
add_embedding_to_chroma(test_user_id, test_embedding)

def euclidean_to_cosine(distance):
    """
    Convert Euclidean distance between normalized vectors to cosine similarity.
    """
    return 1 - (distance ** 2) / 2

# Query function with cosine similarity conversion
def query_embedding_cosine(query_vector, top_k=3):
    if isinstance(query_vector, np.ndarray):
        query_vector = query_vector.tolist()

    norm = np.linalg.norm(query_vector)
    if norm == 0:
        print("❌ Zero-norm embedding.")
        return []

    query_vector = (np.array(query_vector) / norm).tolist()

    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        print("🔍 Query Results (with cosine similarity):")
        for i in range(len(results['ids'][0])):
            dist = results['distances'][0][i]
            cosine_sim = euclidean_to_cosine(dist)
            print(f"🔸 Match: {results['metadatas'][0][i]['name']}, Cosine Similarity: {cosine_sim:.4f}")

        return results

    except Exception as e:
        print(f"❌ Error during query: {e}")
        return []

# Run test query
print("\n🧪 Running test query with cosine similarity...")
query_embedding_cosine(test_embedding, top_k=2)
