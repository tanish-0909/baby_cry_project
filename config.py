import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
N_MELS = 128
MFCC_N = 40

WINDOW_SEC = 2.0
HOP_SEC = 1.0

CRY_THRESHOLD = 0.6 


YOLO_WEIGHTS        = r"final_yolo_weights_hfd\best_weights.pt"
CRY_STATUS_WEIGHTS  = r"final_yolo_weights_hfd\notcry_cry_classifier.pth"
CRY_TYPE_WEIGHTS    = r"final_yolo_weights_hfd\best_infantcry_model.pth"

CRY_TYPE_LABELS = ["Belly Pain", "Burping", "discomfort", "hungry", "tired"]

INPUT_VIDEO1         = r"test_videos\BellyPain_NeedsBurping.mp4"
OUTPUT_VIDEO1        = r"test_outputs\baby_state_output1.mp4"


INPUT_VIDEO3         = r"test_videos\pain_baby cry.mp4"
OUTPUT_VIDEO3        = r"test_outputs\baby_state_output3.mp4"

DATA_ROOT = r"train\listening_beyond_the_cry"

TEST_AUDIO = r"train\tired_249c.wav"