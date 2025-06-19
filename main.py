from utils.utils import process_face, process_card
from db.database import create_table, insert_user, get_all_users


def main(face_path, card_path):
    _ , face_embedding = process_face(face_path)
    card_result = process_card(card_path)
    
    insert_user(
        name = card_result.get("name"),
        father_name = card_result.get("father_name"),
        family_name = card_result.get("family_name"),
        mother_name = card_result.get("mother_name"),
        place_birth = card_result.get("place_birth"),
        image_path = card_path,
        embedding = face_embedding  # Replace with actual embedding if available
    )
    
