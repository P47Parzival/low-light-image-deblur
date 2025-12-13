from roboflow import Roboflow
from ultralytics import YOLO
import os
import sys

def main():
    # 1. Download the BEST dataset (DB Rail)
    # REPLACE with your API Key
    api_key = "u0leKBS1HrAOJli1hvLI"
    
    if api_key == "YOUR_API_KEY":
        print("Please replace 'YOUR_API_KEY' with your actual Roboflow API key in the script.")
        # We continue to show the user what will happen
    
    # Check if dataset already exists to avoid re-downloading
    dataset_location = None
    if os.path.exists("db-rail-1"): # Example default path, logic below is better
        pass

    try:
        rf = Roboflow(api_key=api_key)
        print("Downloading DB Rail Dataset...")
        project = rf.workspace("db-rail").project("train-wagon-cv-project")
        dataset = project.version(3).download("yolov8")
        dataset_location = dataset.location
    except Exception as e:
        print(f"Error downloading dataset (Check API Key): {e}")
        # If we can't download, we can't train, unless user has data locally
        return

    # 2. Train YOLOv8 with "Anti-Catastrophic Failure" settings
    print("Starting Fine-tuning...")
    
    # Load the pre-trained model
    model = YOLO('yolov8n.pt') 

    # Train with safeguards
    results = model.train(
        data=f"{dataset_location}/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project='railway_hackathon',
        name='wagon_counter_v1',
        
        # --- SAFEGUARDS ---
        freeze=10,      # Freeze the first 10 layers (Backbone) to keep pre-trained features
        lr0=0.001,      # Lower initial learning rate (default is usually 0.01)
        optimizer='AdamW', # Gentle optimizer
        patience=10,    # Stop early if not improving
        cos_lr=True,    # Cosine learning rate decay for smooth convergence
    )
    
    print("Training Complete. Best weights saved at: runs/railway_hackathon/wagon_counter_v1/weights/best.pt")

if __name__ == '__main__':
    main()
