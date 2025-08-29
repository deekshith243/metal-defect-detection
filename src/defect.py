import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
import os

# --- Configuration ---
MODEL_PATH = 'faster_rcnn_model.pth' # Make sure this file is in the same directory
NUM_CLASSES = 5
LABEL_MAP = {
    0: "background",
    1: "cut",     # Changed from "scratch"
    2: "dent",
    3: "pit",
    4: "scratch"  # Changed from "cut"
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Helper Functions ---
def get_model_instance_segmentation(num_classes):
    # Load a pre-trained Faster R-CNN model
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one for your task
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def predict_image(image_path, model_path):
    print(f"Processing image: {image_path}")

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predictions = model(image_tensor)

    pred_boxes = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_scores = predictions[0]['scores'].cpu().numpy()

    image_np = np.array(image)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image_np)
    ax.axis('off')

    detected_defects = []

    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score > 0.5:
            xmin, ymin, xmax, ymax = [int(i) for i in box]

            # Create a rectangle patch
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 fill=False, color='red', linewidth=2)
            ax.add_patch(rect)

            # Get the class name
            class_name = LABEL_MAP.get(label, "unknown")
            detected_defects.append(class_name)

            # Add label text
            label_text = f"{class_name}: {score:.2f}"
            ax.text(xmin, ymin - 5, label_text, color='white',
                    bbox=dict(facecolor='red', alpha=0.5))

    # --- Print Final Result ---
    final_status_message = "Status: No defects found."
    if detected_defects:
        final_status_message = "Status: Disc has defects."

    print("\n--- Final Status ---")
    print(final_status_message)

    if detected_defects:
        print(f"Defects found: {', '.join(sorted(list(set(detected_defects))))}")

    final_text = [final_status_message]
    if detected_defects:
        final_text.append(f"Defects found: {', '.join(sorted(list(set(detected_defects))))}")

    y_position = 10
    for line in final_text:
        plt.text(10, y_position, line, color='white', fontsize=12,
                 bbox=dict(facecolor='black', alpha=0.5))
        y_position += 30

    plt.title("Metal Defect Detection")
    plt.show()

if __name__ == '__main__':
    print("Please select an image file to upload:")
    uploaded = files.upload()
    if uploaded:
        uploaded_filename = list(uploaded.keys())[0]
        # Move the uploaded file to the current directory for easy access
        !mv "/content/{uploaded_filename}" .
        predict_image(uploaded_filename, MODEL_PATH)
    else:
        print("No file selected. Prediction aborted.")
