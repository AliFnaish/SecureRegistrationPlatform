import cv2

from utils.face_embedder import extract_face_embedding
from utils.ocr_extractor import extract_fields_from_id
from utils.liveness_checker import detect_liveness

def process_face(face_path):
    face_img = cv2.imread(face_path)
    if face_img is None:
        raise ValueError(f"Image not found or unreadable at: {face_path}")
    face_liveness = detect_liveness(face_img)
    face_embedding = extract_face_embedding(face_img)
    return face_liveness, face_embedding

def process_card(card_path):
    print(f"Processing ID card image: {card_path}")
    fields = extract_fields_from_id(card_path, verbose=True)
    print(f"Extracted fields: {fields}")
    return fields
