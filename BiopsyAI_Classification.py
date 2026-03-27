import os
import sys
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
import glob

# Team Name Configuration
TEAM_NAME = "BiopsyAI"
MODEL_PATH = "models/classification/classifier.pt"

def load_model():
    # Adjusted to your model architecture loading logic
    model = torch.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

def run_inference(test_folder):
    model = load_model()
    results = []
    
    # Standard transforms (Adjust if you used different ones during training)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_paths = glob.glob(os.path.join(test_folder, "*.*"))
    print(f"Found {len(image_paths)} images for classification.")

    for img_path in image_paths:
        try:
            img_id = os.path.basename(img_path)
            img = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(img).unsqueeze(0)
            
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                
            results.append({
                "ImageID": img_id,
                "PredictedLabel": pred.item()
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    df = pd.DataFrame(results)
    output_file = f"{TEAM_NAME}_Classification.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python BiopsyAI_Classification.py <test_folder_path>")
    else:
        run_inference(sys.argv[1])
