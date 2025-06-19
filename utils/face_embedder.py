import numpy as np
from deepface import DeepFace


def extract_face_embedding(face_img):
    """
    Extracts and normalizes the face embedding from the given image path.

    Args:
        face_img (str): Path to the image file containing the face.

    Returns:
        list: Normalized embedding vector, or None on failure.
    """
    
    try:
        embedding = DeepFace.represent(
            img_path=face_img,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]
    
        # Normalize
        face_embedding = np.array(embedding)
        norm = np.linalg.norm(face_embedding)
        if norm > 0:
            face_embedding = face_embedding / norm

        return face_embedding.tolist()
    
    except Exception as e:
        print(f"Failed to extract embedding: {str(e)}")
        return None
