#########   streamlit run app.py


import streamlit as st
import os
import numpy as np
from PIL import Image
from chromadb import Client
from db.database import insert_user
from utils.utils import process_face, process_card
from utils.store_in_chroma import add_embedding_to_chroma

st.set_page_config(page_title="AI Registration System", layout="centered")
st.title("🧠 AI-Powered Registration System")

# Webcam Capture
st.header("1️⃣ Capture Face Image")
face_img = st.camera_input("📷 Capture your face")

# Upload ID card image
st.header("2️⃣ Upload ID Card Image")
id_file = st.file_uploader("Upload scanned ID card", type=["jpg", "jpeg", "png"])

if face_img and id_file:
    # Save images
    os.makedirs("temp", exist_ok=True)
    face_path = "temp/face_cam.jpg"
    id_path = f"temp/{id_file.name}"

    with open(face_path, "wb") as f:
        f.write(face_img.getbuffer())
    with open(id_path, "wb") as f:
        f.write(id_file.getbuffer())

    st.image(face_path, caption="🧍 Captured Face", use_column_width=True)
    st.image(id_path, caption="🪪 ID Card", use_column_width=True)

    if st.button("🔍 Process"):
        st.info("▶️ Step 1: Checking liveness and extracting face embedding...")
        try:
            liveness, embedding = process_face(face_path)
            if not liveness:
                st.error("❌ Liveness check failed.")
                st.stop()
            if embedding is None:
                st.error("❌ Failed to extract face embedding.")
                st.stop()
            st.success("✅ Live face verified and embedding extracted.")
        except Exception as e:
            st.exception(f"Face processing error: {e}")
            st.stop()

        st.info("▶️ Step 2: Extracting ID card fields...")
        try:
            fields = process_card(id_path)
            if not fields or not fields.get("الاسم"):
                st.error("❌ Failed to extract key ID fields.")
                st.stop()
            st.success("✅ OCR extraction successful.")
            st.json(fields)
        except Exception as e:
            st.exception(f"OCR error: {e}")
            st.stop()

        st.info("▶️ Step 3: Inserting into SQLite...")
        try:
            insert_user(
                name=fields.get("الاسم", ""),
                father_name=fields.get("اسم الاب", ""),
                family_name=fields.get("الشهرة", ""),
                mother_name=fields.get("اسم الام وشهرتها", ""),
                place_birth=fields.get("محل الولادة", ""),
                image_path=face_path,
                embedding=str(embedding)
            )
            st.success("✅ Stored in SQLite.")
        except Exception as e:
            st.exception(f"SQLite error: {e}")
            st.stop()

        st.info("▶️ Step 4: Saving embedding in ChromaDB...")
        try:
            add_embedding_to_chroma(fields["الاسم"], embedding)
            st.success("✅ Stored in ChromaDB.")
        except Exception as e:
            st.exception(f"ChromaDB insert error: {e}")
            st.stop()

        st.info("▶️ Step 5: Verifying embedding in ChromaDB...")
        try:
            client = Client(
                chroma_db_impl="duckdb+parquet",
                persist_directory="db/chroma"
            )
            collection = client.get_collection(name="face_embeddings")

            results = collection.query(
                query_embeddings=[embedding],
                n_results=1
            )

            st.write("🎯 Query Result:")
            st.json(results)

            if results["ids"][0][0] == fields["الاسم"]:
                st.success("✅ Verification successful: User matched.")
            else:
                st.error("❌ Verification failed: No match.")
        except Exception as e:
            st.exception(f"ChromaDB query error: {e}")
