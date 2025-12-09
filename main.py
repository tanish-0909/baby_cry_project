import torch
import os
from ultralytics import YOLO

# Import local modules
from config import *
from utils import *

def main():
    # 1. Setup device and ensure folders exist
    print(f"Running on: {DEVICE}")
    os.makedirs(os.path.dirname(OUTPUT_VIDEO1), exist_ok=True)

    # 2. Load Models
    print("Loading models...")
    
    # YOLO
    yolo = YOLO(YOLO_WEIGHTS)

    # Cry Status (Binary)
    cry_status_model = CryClassifier().to(DEVICE)
    cry_status_model.load_state_dict(torch.load(CRY_STATUS_WEIGHTS, map_location=DEVICE))
    cry_status_model.eval()

    # Cry Type (Multiclass) - Feature Extractor & Fusion
    crytype_ckpt = torch.load(CRY_TYPE_WEIGHTS, map_location=DEVICE)
    
    cry_type_mel = ResNetMel(out_dim=512).to(DEVICE)
    cry_type_mel.load_state_dict(crytype_ckpt["cnn"])
    cry_type_mel.eval()

    cry_type_model = FusionClassifier(emb_dim=512, mfcc_dim=MFCC_N*2, num_classes=5).to(DEVICE)
    cry_type_model.load_state_dict(crytype_ckpt["fusion"])
    cry_type_model.eval()

    # 3. Process Videos
    # We pass all models explicitly to avoid NameErrors in utils.py
    
    if os.path.exists(INPUT_VIDEO1):
        process_video(INPUT_VIDEO1, OUTPUT_VIDEO1, yolo, cry_status_model, cry_type_mel, cry_type_model)
        print("DONE! Saved:", OUTPUT_VIDEO1)
    else:
        print(f"Warning: {INPUT_VIDEO1} not found.")

    if os.path.exists(INPUT_VIDEO3):
        process_video(INPUT_VIDEO3, OUTPUT_VIDEO3, yolo, cry_status_model, cry_type_mel, cry_type_model)
        print("DONE! Saved:", OUTPUT_VIDEO3)

if __name__ == "__main__":
    main()