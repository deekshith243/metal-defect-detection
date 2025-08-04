# src/train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import MetalDefectDataset, get_transform
from src.model import get_model_instance_segmentation
from src.config import *
import matplotlib.pyplot as plt

def collate_fn(batch):
    return tuple(zip(*batch))

def train_model():
    # Load model and move to device
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(DEVICE)

    # Load datasets
    train_dataset = MetalDefectDataset(IMAGES_DIR_TRAIN, ANNOTATIONS_TRAIN, transforms=get_transform())
    val_dataset = MetalDefectDataset(IMAGES_DIR_VAL, ANNOTATIONS_VAL, transforms=get_transform())
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("Starting Training...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            if (i+1) % 10 == 0:
                print(f"Epoch: {epoch+1}, Batch: {i+1}, Loss: {losses.item():.4f}")

        # Update learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()
            
        print(f"Epoch {epoch+1} finished. Average loss: {epoch_loss / len(train_loader):.4f}")
        
    print("Training finished.")
    
    # Save the trained model
    torch.save(model.state_dict(), 'faster_rcnn_model.pth')
    print("Model saved as faster_rcnn_model.pth")

if __name__ == '__main__':
    train_model()