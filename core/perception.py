
# using yolo v8 nano here 

import cv2
import numpy as np
from ultralytics import YOLO

class PerceptionModule:
    def __init__(self, model_path='yolov8n.pt'):
        print(f"Loading YOLOv8 model from {model_path}...")
        # This will automatically download yolov8n.pt if it's not in your folder
        self.model = YOLO(model_path)
        
        # COCO dataset class IDs we care about for the ADAS challenge
        # Expanded COCO dataset classes to include generic road obstacles
        self.target_classes = {
            0: 'pedestrian',
            1: 'bicycle',       # Obstacle
            2: 'car',
            3: 'motorcycle',    # Obstacle
            5: 'bus',
            7: 'truck',
            9: 'traffic light', # Static obstacle
            10: 'fire hydrant', # Static obstacle
            11: 'stop sign',
            13: 'bench'         # Static obstacle
        }

    def process_frame(self, frame):
        """
        Takes a BGR frame, runs inference, and returns the annotated frame
        along with a structured list of detections for the Safety/Planning modules.
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)[0]
        
        detections = []
        annotated_frame = frame.copy()

        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Filter only the classes we need and ensure confidence is decent
            if class_id in self.target_classes and conf > 0.4:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{self.target_classes[class_id]} {conf:.2f}"
                
                # Store detection data for later modules (like AEB and stopping logic)
                detections.append({
                    'class': self.target_classes[class_id],
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })
                
                # Draw bounding box based on object type
                color = (0, 0, 255) if class_id == 0 else (0, 255, 0) # Red for pedestrians, Green for others
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated_frame, detections