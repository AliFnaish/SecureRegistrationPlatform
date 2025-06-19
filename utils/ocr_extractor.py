import os
import easyocr
import re

# Avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def extract_fields_from_id(image_path, verbose=False):
    try:
        if verbose:
            print(f"📷 Processing Image: {image_path}\n")

        print(1)
        reader = easyocr.Reader(['ar'], gpu=False)
        # Get OCR lines (text only) — NO unpacking
        ocr_lines,a = reader.readtext(image_path, detail=0)
        
        print(a)

        if verbose:
            print("📝 OCR Lines:")
            for line in ocr_lines:
                print(f"→ {line}")

        fields = {
            "الاسم": "",
            "الشهرة": "",
            "اسم الاب": "",
            "اسم الام وشهرتها": "",
            "محل الولادة": ""
        }

        lines = [line.strip() for line in ocr_lines]

        def find_field(keyword):
            for i, line in enumerate(lines):
                if keyword in line:
                    parts = re.split(rf"{keyword}[:\s]*", line, maxsplit=1)
                    if len(parts) > 1 and parts[1].strip():
                        return parts[1].strip()
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        if not re.match(r"^(اسم|عبطبي|3مد)$", next_line):
                            return next_line.strip()
            return ""

        fields["الاسم"] = find_field("الاسم")
        fields["الشهرة"] = find_field("الشهرة")
        fields["اسم الاب"] = find_field("الاب")
        fields["اسم الام وشهرتها"] = find_field("الام وشهرتها")
        fields["محل الولادة"] = find_field("محل الولادة")

        if not fields["الشهرة"]:
            surname_parts = []
            for line in lines:
                if line.startswith("الشهرا") or line.startswith("رة :"):
                    surname_parts.append(line.replace("الشهرا", "").replace("رة :", "").strip())
            if surname_parts:
                fields["الشهرة"] = " ".join(surname_parts).strip()

        def clean_field(text):
            cleaned = re.sub(r"\b(اسم|عبطبي|3مد|بطاقة|توفيع صاحب العلاقة|ديا ب)\b", "", text)
            return " ".join(cleaned.split())

        for key in fields:
            fields[key] = clean_field(fields[key])

        if verbose:
            print("\n🔍 Final Extracted Fields:")
            for k, v in fields.items():
                print(f"{k}: {v if v else '❌ Not Detected'}")

        return fields
    except Exception as e:
        print(f"❌ Error during OCR extraction: {e}")
        return None
    
if __name__ == "__main__":
    extract_fields_from_id(r"assets\id_sobhi.jpg", verbose=True)
