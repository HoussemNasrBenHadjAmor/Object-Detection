from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO 
from pydantic import BaseModel
from typing import List
import cv2
import numpy as np
import io
from starlette.responses import StreamingResponse


app = FastAPI()

origins = [
    'http://localhost:3000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins= origins,
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*'],
)

@app.get('/')
async def read_root():
    return {'Hello':'Word'}

# Charge our model
model = YOLO('C:/Users/Houssem/Desktop/pytorch/backend/best.pt')

class Detection(BaseModel):
    box: List[float]
    confidence: float
    class_id: int

# Define class names
class_names = {
    3: "Referee",
    2: "Player",
    1: "Goalkeeper",
    0: "Ball"
}

@app.post('/detect/', response_model=List[Detection])
async def detect_object(file: UploadFile):
    # Process the uploaded image for object detection
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
    # Perform object detection with YOLOv9c
    results = model.predict(image)
    detections = results[0].boxes.data.cpu().numpy()
    
    # Convert predictions to numpy array
    response = []
    for detection in detections:
        box = detection[:4].tolist()  # [x1, y1, x2, y2]
        confidence = float(detection[4])
        class_id = int(detection[5])
        response.append(Detection(box=box, confidence=confidence, class_id=class_id))   

     # Draw detections on the image
    image_with_detections = draw_detections(image, response)    

    # Encode image back to bytes
    _, img_encoded = cv2.imencode('.jpg', image_with_detections)
    image_bytes = img_encoded.tobytes()
    
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/jpeg")


def draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    
    color_mapping = { 
        0: (0, 255, 0),  # Lime referee
        1: (255, 0, 0),  # Red for player
        2: (0, 0, 255), # Blue for goalkeeper
        3:(255,255,0) # Yellow for ball
    }

    for detection in detections:
        bbox = detection.box
        color = color_mapping.get(detection.class_id, (0, 255, 255))  # Default to yellow if class not in mapping
        class_name = class_names.get(detection.class_id, "Unknown")

        # Draw bounding box
        cv2.rectangle(image, (int(bbox[0]),int (bbox[1])) , (int (bbox[2]),int (bbox[3])) , color, 2)
        # Prepare label with class name and confidence
        label = f"{class_name}: {detection.confidence:.2f}"
        # Get label size
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # Draw label background
        cv2.rectangle(image, (int (bbox[0]), int (bbox[1]) - h - 10), (int (bbox[0]) + w, int (bbox[1])), color , -1)
        # Put label on image
        cv2.putText(image, label, (int (bbox[0]), int (bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return image

    


