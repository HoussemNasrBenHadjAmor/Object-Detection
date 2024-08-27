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
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F

# Assuming your script is in the root directory
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.create_fasterrcnn_model import create_model


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



def draw_detections(image: np.ndarray, detections: List[Detection], class_names) -> np.ndarray:
    
    color_mapping = { 
        0: (0, 255, 0),  # Lime referee
        1: (255, 0, 0),  # Red for player
        2: (0, 0, 255),  # Blue for goalkeeper
        3: (255, 255, 0) # Yellow for ball
    }

    height, width = image.shape[:2]

    for detection in detections:
        bbox = detection.box
        color = color_mapping.get(detection.class_id, (0, 255, 255))  # Default to yellow if class not in mapping
        class_name = class_names.get(detection.class_id, "Unknown")

        # Draw bounding box
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        
        # Prepare label with class name and confidence
        label = f"{class_name}: {detection.confidence:.2f}"
        # Get label size
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # Calculate label background position
        label_bg_x1 = int(bbox[0])
        label_bg_y1 = int(bbox[1]) - h - 10
        label_bg_x2 = label_bg_x1 + w
        label_bg_y2 = int(bbox[1])

        # Adjust position if label background is outside the image boundaries
        if label_bg_x1 < 0:
            label_bg_x1 = 0
            label_bg_x2 = w
        if label_bg_y1 < 0:
            label_bg_y1 = 0
            label_bg_y2 = h + 10
        if label_bg_x2 > width:
            label_bg_x2 = width
            label_bg_x1 = label_bg_x2 - w
        if label_bg_y2 > height:
            label_bg_y2 = height
            label_bg_y1 = label_bg_y2 - (h + 10)

        # Draw label background
        cv2.rectangle(image, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), color, -1)
        # Put label on image
        text_x = label_bg_x1
        text_y = label_bg_y2 - 5

        cv2.putText(image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return image


@app.post('/detect/yolov9', response_model=List[Detection])
async def detect_object(file: UploadFile, threshold: float = 0.25 ):
    # Load our yolo model
    #MODEL_PATH = 'C:/Users/Houssem/Desktop/pytorch/backend_v2/last.pt'
    MODEL_PATH = 'C:/Users/Houssem/Desktop/pytorch/backend_v2/yolov9_fgsm_aug_40_epochs.pt'
    model = YOLO(MODEL_PATH)

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
        if confidence >= threshold:
            response.append(Detection(box=box, confidence=confidence, class_id=class_id))  

    # Define class names
    class_names = {
        0: "Ball",
        1: "Goalkeeper",
        2: "Player",
        3: "Referee",
        }     

     # Draw detections on the image
    image_with_detections = draw_detections(image, response, class_names)    

    # Encode image back to bytes
    _, img_encoded = cv2.imencode('.jpg', image_with_detections)
    image_bytes = img_encoded.tobytes()
    
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/jpeg")



@app.post('/detect/yolov8', response_model=List[Detection])
async def detect_object(file: UploadFile, threshold: float = 0.25 ):
    # Load our yolo model
    MODEL_PATH = 'C:/Users/Houssem/Desktop/pytorch/backend_v2/yolov9_100_epochs.pt'
    model = YOLO(MODEL_PATH)

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
        if confidence >= threshold:
            response.append(Detection(box=box, confidence=confidence, class_id=class_id))  

    # Define class names
    class_names = {
        0: "Ball",
        1: "Goalkeeper",
        2: "Player",
        3: "Referee",
        }     

     # Draw detections on the image
    image_with_detections = draw_detections(image, response, class_names)    

    # Encode image back to bytes
    _, img_encoded = cv2.imencode('.jpg', image_with_detections)
    image_bytes = img_encoded.tobytes()
    
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/jpeg")




    
@app.post('/detect/ssd', response_model=List[Detection])
async def detect_object(file: UploadFile, threshold:float = 0.25):
    # Model path 
    MODEL_PATH = 'C:/Users/Houssem/Desktop/pytorch/backend_v2/ssd.pth'
    
    # Load the state_dict
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    # Get the classes and classes number from the checkpoint
    NUM_CLASSES = checkpoint['config']['NC']

    # Create our model
    model = ssd300_vgg16(num_classes = NUM_CLASSES, pretrained=False)

    # Load the state_dict into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to device and evaluation mode
    model.to(device).eval()

    # Process the uploaded image for object detection
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Convert the image to a PyTorch tensor and normalize it
    image_tensor = F.to_tensor(image).to(device)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
 
    # Perform object detection with Faster R-CNN
    with torch.no_grad():
        results = model(image_tensor)
        print(f'length of results : {len(results[0]['labels'])}')

    detections = results[0]
    response = []
    for box, score, label in zip(detections['boxes'], detections['scores'], detections['labels']):
         if score >= threshold:  # Filter out low-confidence detections
            response.append(Detection(
                box=box.cpu().numpy().tolist(),
                confidence=score.item(),
                class_id=label.item()
            ))

    # Define class names
    class_names = {
        0: "_background_",
        1: "Ball",
        2: "Goalkeeper",
        3: "Player",
        4: 'Referee'
        }     

    # Draw detections on the image
    image_with_detections = draw_detections(image, response, class_names)

    # Encode image back to bytes
    _, img_encoded = cv2.imencode('.jpg', image_with_detections)
    image_bytes = img_encoded.tobytes()

    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/jpeg")



@app.post('/detect/frcnn', response_model=List[Detection])
async def detect_object(file: UploadFile, threshold:float = 0.75):
    # Model path 
    MODEL_PATH = 'C:/Users/Houssem/Desktop/pytorch/backend_v2/frcnn_fgsm_aug_40_epochs.pth'

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

    # Convert the image to a PyTorch tensor and normalize it
    image_tensor = F.to_tensor(image).to(device)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
 
    # Perform object detection with Faster R-CNN
    with torch.no_grad():
        results = model(image_tensor)

    detections = results[0]
    response = []
    for box, score, label in zip(detections['boxes'], detections['scores'], detections['labels']):
         if score >= threshold:  # Filter out low-confidence detections
            response.append(Detection(
                box=box.cpu().numpy().tolist(),
                confidence=score.item(),
                class_id=label.item()
            ))

    # Define class names
    class_names = {
        0: "_background_",
        1: "Ball",
        2: "Goalkeeper",
        3: "Player",
        4: 'Referee'
        }     

    # Draw detections on the image
    image_with_detections = draw_detections(image, response, class_names)

    # Encode image back to bytes
    _, img_encoded = cv2.imencode('.jpg', image_with_detections)
    image_bytes = img_encoded.tobytes()

    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/jpeg")