from ultralytics import YOLO
model = YOLO("liveness_model_yolov8n.pt")

def detect_liveness(face_img):
    """
    Detects liveness from the given face image path.
    
    Args:
        face_path (str): Path to the image file containing the face.
        
    Returns:
        bool: True if the face is live, False otherwise.
    """
    
    print("Checking liveness for face ...")
    
    try:
        
        results = model.predict(source = face_img,  save = True)
        
        for box in results[0].boxes:
            cls_id = int(box.cls)
            confidence = float(box.conf)
            label = results[0].names[cls_id]
        
            print(f"Detected label: {label}, Confidence: {confidence:.2f}")
            if label == "Real" and confidence > 0.5:
                print("Liveness check passed.")
                return True
            else:
                print("Liveness check failed.")
                return False

    except Exception as e:
        print(f"⚠️ Error during liveness detection: {str(e)}")
        return False