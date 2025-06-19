import numpy as np
from deepface import DeepFace
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB client with persistence folder
client = chromadb.Client(Settings(
    persist_directory="db/chroma",
    anonymized_telemetry=False
))

# Create or get the face embeddings collection
collection = client.get_or_create_collection(name="face_embeddings")

def extract_embedding(face_img_path):
    """
    Extract and normalize face embedding from image path.
    """
    try:
        embedding = DeepFace.represent(
            img_path=face_img_path,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

        face_embedding = np.array(embedding)
        norm = np.linalg.norm(face_embedding)
        if norm > 0:
            face_embedding = face_embedding / norm

        return face_embedding.tolist()

    except Exception as e:
        print(f"Failed to extract embedding: {e}")
        return None

def store_embedding(user_id, embedding):
    """
    Store embedding in ChromaDB collection.
    """
    if not user_id or not embedding:
        print("❌ Invalid user ID or embedding.")
        return

    collection.add(
        ids=[user_id],
        embeddings=[embedding],
        metadatas=[{"name": user_id}]
    )
    print(f"✅ Stored embedding for user: {user_id}")

def print_all_embeddings():
    """
    Print all stored embeddings metadata and partial embedding vectors.
    """
    results = collection.get(include=["embeddings", "metadatas"])

    print("📦 Stored Embeddings:")
    for i, emb_id in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        emb = results["embeddings"][i]
        print(f"{i+1}. ID: {emb_id}")
        print(f"   Metadata: {meta}")
        print(f"   Embedding (first 5): {emb[:5]}")
        print("--------------------------------------------------")

if __name__ == "__main__":
    # Example usage
    user_id = "ali_karar"  # your chosen unique user ID
    face_image_path = r"C:\Users\aaa_f\Project\Complete_System(v3)\assets\id_card.jpg"  # your image path

    embedding = extract_embedding(face_image_path)
    if embedding is not None:
        store_embedding(user_id, embedding)

    print_all_embeddings()
