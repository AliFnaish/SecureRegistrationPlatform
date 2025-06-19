import sys
import numpy as np
from chromadb import Client
from db.database import insert_user
from utils.utils import process_face, process_card
from utils.store_in_chroma import add_embedding_to_chroma

def main():
    face_img_path = r"assets\face.jpg"
    card_img_path = r"assets\id_sobhi.jpg"

    print("▶️ Step 1: Process face image (liveness + embedding)")
    try:
        face_liveness, face_embedding = process_face(face_img_path)
        if not face_liveness:
            print("❌ Liveness check failed. Aborting.")
            return
        if face_embedding is None:
            print("❌ Failed to extract face embedding. Aborting.")
            return
        print("✅ Face processed successfully.")
    except Exception as e:
        print(f"❌ Error processing face image: {e}")
        return

    print("\n▶️ Step 2: Process ID card image (OCR extraction)")
    try:
        card_fields = process_card(card_img_path)
        if not card_fields:
            print("❌ Failed to extract ID card data. Aborting.")
            return
        print("✅ ID card data extracted:")
        for k, v in card_fields.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"❌ Error processing ID card image: {e}")
        return

    user_id = card_fields.get("الاسم")
    if not user_id:
        print("❌ User ID (الاسم) not found in extracted fields. Aborting.")
        return

    print("\n▶️ Step 3: Store user data into SQLite DB")
    try:
        insert_user(
            name=card_fields.get("الاسم", ""),
            father_name=card_fields.get("اسم الاب", ""),
            family_name=card_fields.get("الشهرة", ""),
            mother_name=card_fields.get("اسم الام وشهرتها", ""),
            place_birth=card_fields.get("محل الولادة", ""),
            image_path=face_img_path,
            embedding=str(face_embedding.tolist())
        )
        print("✅ User data stored in SQLite DB.")
    except Exception as e:
        print(f"❌ Failed to insert user into DB: {e}")
        return

    print("\n▶️ Step 4: Store face embedding in ChromaDB")
    try:
        add_embedding_to_chroma(user_id, face_embedding)
    except Exception as e:
        print(f"❌ Failed to store embedding in ChromaDB: {e}")
        return

    print("\n▶️ Step 5: Query ChromaDB to verify embedding")
    try:
        client = Client(
            chroma_db_impl="duckdb+parquet",
            persist_directory="db/chroma"
        )
        collection = client.get_collection(name="face_embeddings")

        results = collection.query(
            query_embeddings=[face_embedding.tolist()],
            n_results=1
        )

        print("🎯 Query Results:")
        print(results)

        if results and results["ids"] and results["ids"][0][0] == user_id:
            print("✅ Verification successful: Matching user found in DB.")
        else:
            print("❌ Verification failed: No matching user found.")
    except Exception as e:
        print(f"❌ Error querying ChromaDB: {e}")

if __name__ == "__main__":
    main()
