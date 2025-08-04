# main.py

import os
from src.train import train_model

def main():
    print("Welcome to the Metal Defect Detection Training Script.")
    
    # Check if a trained model exists
    model_file = 'faster_rcnn_model.pth'
    if os.path.exists(model_file):
        choice = input("A trained model already exists. Do you want to retrain? (y/n): ")
        if choice.lower() != 'y':
            print("Training aborted.")
            return
            
    print("Starting model training...")
    train_model()
    
    print("Training finished. The trained model is saved as 'faster_rcnn_model.pth'.")

if __name__ == '__main__':
    main()