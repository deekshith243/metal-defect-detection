# src/inference.py

import torch
import cv2
import numpy as np
from torchvision.transforms import functional as F
from src.model import get_model_instance_segmentation
from src.config import *
from picamera2 import Picamera2
from libcamera import controls
import time
from PIL import Image

def draw_results(frame, boxes, labels, scores, circles_info):
    """
    Draws bounding boxes for defects and circle info on the frame.
    """
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # Confidence threshold
            xmin, ymin, xmax, ymax = [int(i) for i in box]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            
            # Find the class name from the LABEL_MAP
            class_name = [name for name, idx in LABEL_MAP.items() if idx == label][0]
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(frame, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Draw circle information
    if circles_info:
        for (x, y, r) in circles_info:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

def check_diameter_with_hough(image):
    """
    Detects circles using Hough Transform and checks if their diameters are in range.
    """
    # Convert image to grayscale for Hough Transform
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Use a median blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=100,
        param1=50, 
        param2=30, 
        minRadius=50, 
        maxRadius=200
    )
    
    valid_inner, valid_outer = "Not Valid", "Not Valid"
    detected_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        detected_circles = [(c[0], c[1], c[2]) for c in circles[0, :]]
        
        # Sort circles by radius to easily find inner and outer
        radii = [c[2] for c in circles[0, :]]
        radii.sort()
        
        if len(radii) >= 2:
            inner_radius_pixel = radii[-2]
            outer_radius_pixel = radii[-1]
            
            inner_diameter_cm = 2 * inner_radius_pixel * PIXEL_TO_CM_RATIO
            outer_diameter_cm = 2 * outer_radius_pixel * PIXEL_TO_CM_RATIO
            
            if INNER_DIAMETER_RANGE[0] <= inner_diameter_cm <= INNER_DIAMETER_RANGE[1]:
                valid_inner = "Valid"
            if OUTER_DIAMETER_RANGE[0] <= outer_diameter_cm <= OUTER_DIAMETER_RANGE[1]:
                valid_outer = "Valid"
                
    return detected_circles, valid_inner, valid_outer

def run_realtime_inference(model_path):
    """
    Sets up the Raspberry Pi camera and runs real-time inference.
    """
    print("Initializing Raspberry Pi Camera...")
    picam2 = Picamera2()
    # Configure camera stream
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    picam2.start()

    print("Loading model...")
    # Load model and move to device (CPU for Raspberry Pi)
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print("Starting real-time detection. Press 'q' to quit.")

    while True:
        # Capture frame
        frame = picam2.capture_array()
        
        # Convert the frame from BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- Part 1: Defect Detection ---
        # Convert numpy array to PIL Image, then to tensor for the model
        image_pil = Image.fromarray(frame_rgb)
        image_tensor = F.to_tensor(image_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            predictions = model(image_tensor)
        
        # Extract defect detection results
        pred_boxes = predictions[0]['boxes'].cpu().numpy()
        pred_labels = predictions[0]['labels'].cpu().numpy()
        pred_scores = predictions[0]['scores'].cpu().numpy()
        
        # Get the names of the detected defects
        detected_defects = []
        for label, score in zip(pred_labels, pred_scores):
            if score > 0.5:
                class_name = [name for name, idx in LABEL_MAP.items() if idx == label][0]
                detected_defects.append(class_name)
        
        # Display defect status on frame
        if detected_defects:
            unique_defects = sorted(list(set(detected_defects)))
            defect_text = "Defects Found: " + ", ".join(unique_defects)
            cv2.putText(frame, "Status: Disc has defects.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, defect_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Status: No defects found.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- Part 2: Diameter Measurement ---
        circles, valid_inner, valid_outer = check_diameter_with_hough(frame_rgb)

        # Display diameter status on frame
        cv2.putText(frame, f"Inner Diameter: {valid_inner}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Outer Diameter: {valid_outer}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw results on the frame
        draw_results(frame, pred_boxes, pred_labels, pred_scores, circles)

        # Display the frame
        cv2.imshow("Real-Time Metal Defect Detection", frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model_file_path = 'faster_rcnn_model.pth'
    if os.path.exists(model_file_path):
        run_realtime_inference(model_file_path)
    else:
        print("Please train the model first by running main.py.")