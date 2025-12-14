from roboflow import Roboflow
from ultralytics import YOLO
import os
import shutil
import yaml
from glob import glob

def remap_and_copy(source_path, dest_path_images, dest_path_labels, target_class_id):
    """
    Copies images and creates remapped label files.
    """
    os.makedirs(dest_path_images, exist_ok=True)
    os.makedirs(dest_path_labels, exist_ok=True)

    # Process images and corresponding labels
    # YOLOv8 datasets usually have 'images' and 'labels' folders
    
    # We look for images in the source
    source_images = glob(os.path.join(source_path, 'images', '*'))
    
    for img_path in source_images:
        # Copy Image
        shutil.copy(img_path, dest_path_images)
        
        # Process Label
        basename = os.path.basename(img_path)
        name_root, _ = os.path.splitext(basename)
        label_file = os.path.join(source_path, 'labels', f"{name_root}.txt")
        
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Replace original class_id with target_class_id
                    # We keep coordinates/dimensions same
                    new_line = f"{target_class_id} " + " ".join(parts[1:])
                    new_lines.append(new_line)
            
            if new_lines:
                with open(os.path.join(dest_path_labels, f"{name_root}.txt"), 'w') as f:
                    f.write('\n'.join(new_lines))

def main():
    api_key = "u0leKBS1HrAOJli1hvLI"
    
    if api_key == "YOUR_API_KEY":
        print("Please replace 'YOUR_API_KEY' within the script.")
        return

    # MAPPING CONFIGURATION
    # (Workspace, Project, Version, Target_Class_ID, Target_Class_Name)
    dataset_config = [
        # GLOBAL CLASS 0: Wagon
        ("aispry-ob85t", "wagon-detection-zsnyn", 2, 0, "Wagon"),
        ("alisha-nyb7f", "wagon-detection-qxlxh", 1, 0, "Wagon"),
        ("wagons-thdfd", "cv-alt", 2, 0, "Wagon"),
        
        # GLOBAL CLASS 1: Wagon parts
        ("db-rail", "train-wagon-cv-project", 3, 1, "Wagon parts"),
        
        # GLOBAL CLASS 2: Wagon numbers
        ("sedykh-marat-dxrw3", "wagon-numbers-detection", 1, 2, "Wagon numbers"),
        ("student-ih3dc", "wagon-detection-qc7bh", 1, 2, "Wagon numbers"),
    ]

    rf = Roboflow(api_key=api_key)
    
    # Create Merged Dataset Structure
    MERGED_DIR = "railway_hackathon_merged"
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(MERGED_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(MERGED_DIR, split, 'labels'), exist_ok=True)

    print("-" * 60)
    print("STEP 1: Downloading & Merging Datasets")
    print("-" * 60)

    for workspace, project_id, version, target_id, target_name in dataset_config:
        try:
            print(f"Processing {workspace}/{project_id} v{version} -> Class {target_id} ({target_name})")
            project = rf.workspace(workspace).project(project_id)
            dataset = project.version(version).download("yolov8")
            
            location = dataset.location
            
            # Merge Train, Valid, Test splits
            for split in ['train', 'valid', 'test']:
                # Some datasets might use 'train' or 'valid' folders differently, standardizing here
                src_split_path = os.path.join(location, split)
                if not os.path.exists(src_split_path): 
                    # Try fallback if Roboflow structure varies
                    continue
                    
                dest_images = os.path.join(MERGED_DIR, split, 'images')
                dest_labels = os.path.join(MERGED_DIR, split, 'labels')
                
                remap_and_copy(src_split_path, dest_images, dest_labels, target_id)
                
        except Exception as e:
            print(f"Skipping {project_id}: {e}")

    # Create Custom data.yaml
    yaml_content = {
        'train': os.path.abspath(os.path.join(MERGED_DIR, 'train', 'images')),
        'val': os.path.abspath(os.path.join(MERGED_DIR, 'valid', 'images')),
        'test': os.path.abspath(os.path.join(MERGED_DIR, 'test', 'images')),
        'nc': 3,
        'names': ['Wagon', 'Wagon parts', 'Wagon numbers']
    }
    
    yaml_path = os.path.join(MERGED_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)

    print("-" * 60)
    print(f"Dataset Merged at: {MERGED_DIR}")
    print("STEP 2: Starting Training on Merged Dataset")
    print("-" * 60)

    # Train
    model = YOLO('yolov8n.pt') 
    
    try:
        results = model.train(
            data=yaml_path,
            epochs=50,
            imgsz=640,
            batch=16,
            project='railway_hackathon_take2',
            name='merged_model_v1',
            freeze=10,
            lr0=0.001,
            patience=10
        )
        print("Training Complete!")
    except Exception as e:
        print(f"Training Failed: {e}")

if __name__ == '__main__':
    main()
