from utils.ocr_extractor import extract_fields_from_id

if __name__ == "__main__":
    # Replace this path with your actual ID card image path
    test_image_path = r"C:\Users\aaa_f\Project\Complete_System(v3)\assets\id_card.jpg"

    try:
        extracted_data = extract_fields_from_id(test_image_path)
        print("\n✅ OCR Extraction successful.")
    except FileNotFoundError as fnf_err:
        print(f"❌ File error: {fnf_err}")
    except Exception as e:
        print(f"❌ Error during OCR extraction: {e}")
