import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import glob

# Team Name Configuration
TEAM_NAME = "BiopsyAI"
MODEL_PATH = "models/segmentation/segmentor.pt"

def load_model():
    # Loading as TorchScript (JIT) since segmentor.pt is one
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model

def run_segmentation(test_folder):
    model = load_model()
    output_dir = "predicted_masks"
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(test_folder, "*.*"))
    print(f"Found {len(image_paths)} images for segmentation.")

    for img_path in image_paths:
        try:
            # Original image loading for size reference
            orig_img = cv2.imread(img_path)
            orig_h, orig_w = orig_img.shape[:2]
            
            # Preprocessing (Adjust to match your training)
            img = cv2.resize(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), (224, 224))
            tensor = torch.from_numpy(img).permute(2,0,1).float().div(255.0).unsqueeze(0)
            
            with torch.no_grad():
                # Inference
                output = model(tensor)
                # Apply sigmoid if necessary (assuming raw logits or 0-1)
                prob = torch.sigmoid(output).squeeze().numpy()
                mask = (prob > 0.5).astype(np.uint8) * 255
            
            # Resize back to original dimensions
            final_mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # Save as PNG
            mask_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
            cv2.imwrite(os.path.join(output_dir, mask_name), final_mask)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Finished. Masks saved in '{output_dir}/'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python BiopsyAI_Segmentation.py <test_folder_path>")
    else:
        run_segmentation(sys.argv[1])
