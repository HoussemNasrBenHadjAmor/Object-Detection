from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO 
from pydantic import BaseModel
from typing import List
import cv2
import numpy as np
import io
from starlette.responses import StreamingResponse
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

app = FastAPI()

origins = [
    'http://localhost:3000',
    'http://127.0.0.1:8000'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins= origins,
    allow_credentials = True,
    allow_methods = ['*'],
    allow_headers = ['*'],
)

use_cuda=True
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


@app.get('/')
async def read_root():
    return {'Hello':'Word'}


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


@app.post('/detect/yolov9', response_model=List[Detection])
async def detect_object(file: UploadFile):
    # Load our yolo model
    model = YOLO('C:/Users/Houssem/Downloads/Object Detection/YOLO/YOLOV9_100_epochs/weights/best.pt')

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




    
@app.post('/detect/frcnn', response_model=List[Detection])
async def detect_object(file: UploadFile):

    # Model path 
    MODEL_PATH = 'C:/Users/Houssem/Desktop/pytorch/backend_v2/best_model.pth'

    # Load the state_dict
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Get the classes and classes number from the checkpoint
    NUM_CLASSES = checkpoint['config']['NC']
    #CLASSES = checkpoint['config']['CLASSES']

    # Create our model
    model = fasterrcnn_resnet50_fpn_v2(num_classes = NUM_CLASSES, pretrained=False, coco_model=False)

    # Load the state_dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to device and evaluation mode
    model.to(device).eval()

    # Process the uploaded image for object detection
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
    # Perform object detection with YOLOv9c
    results = model.predict(image)
    print(f'results : {results}')
    #detections = results[0].boxes.data.cpu().numpy()
    
    # Convert predictions to numpy array
    #response = []
    #for detection in detections:
        #box = detection[:4].tolist()  # [x1, y1, x2, y2]
        #confidence = float(detection[4])
        #class_id = int(detection[5])
        #response.append(Detection(box=box, confidence=confidence, class_id=class_id))   

     # Draw detections on the image
    #image_with_detections = draw_detections(image, response)    

    # Encode image back to bytes
    #_, img_encoded = cv2.imencode('.jpg', image_with_detections)
    #image_bytes = img_encoded.tobytes()
    
    #return StreamingResponse(io.BytesIO(image_bytes), media_type="image/jpeg")

