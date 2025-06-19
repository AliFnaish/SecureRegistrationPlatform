import chromadb
from chromadb.config import Settings
import numpy as np

# Initialize ChromaDB client with persistence
client = chromadb.Client(Settings(
    persist_directory="db/chroma",  # update path as needed
    anonymized_telemetry=False
))

# Create or get collection
collection = client.get_or_create_collection(name="face_embeddings")

def store_embedding(user_id, embedding):
    """
    Stores a normalized face embedding if it doesn't already exist.
    """
    try:
        # Normalize embedding before storing
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = (np.array(embedding) / norm).tolist()
        else:
            print("❌ Zero-norm embedding.")
            return

        existing = collection.get(ids=[user_id])
        if existing and existing['ids']:
            print(f"⚠️ User '{user_id}' already exists. Skipping insertion.")
            return

        collection.add(
            ids=[user_id],
            embeddings=[embedding],
            metadatas=[{"name": user_id}]
        )

        print(f"✅ Stored embedding for user: {user_id}")
    
    except Exception as e:
        print(f"❌ Error while storing embedding: {e}")

def add_embedding_to_chroma(user_id, face_embedding):
    """
    Adds a validated embedding to ChromaDB.
    """
    if not user_id or face_embedding is None:
        print("❌ Invalid user ID or embedding.")
        return

    # Normalize the embedding before passing it to store_embedding
    if isinstance(face_embedding, np.ndarray):
        face_embedding = face_embedding.tolist()
    elif isinstance(face_embedding, list):
        norm = np.linalg.norm(face_embedding)
        if norm > 0:
            face_embedding = (np.array(face_embedding) / norm).tolist()
        else:
            print("❌ Zero-norm embedding.")
            return
    else:
        print("❌ Unsupported embedding format.")
        return

    store_embedding(user_id, face_embedding)
