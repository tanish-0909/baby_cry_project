import torch
from torch import nn
import numpy as np
import librosa
import soundfile as sf
import cv2
import moviepy.editor as mp
import torchvision.models as models
import os
from config import *

# --- MODEL DEFINITIONS ---
class CryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x)).squeeze()

class ResNetMel(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        # Use weights=None explicitly as requested
        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base = nn.Sequential(*list(base.children())[:-1])
        self.embedding_dim = base.fc.in_features
        self.head = nn.Sequential(
            nn.Linear(self.embedding_dim, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = self.base(x)               
        x = x.reshape(x.size(0), -1)  
        return self.head(x)       

class FusionClassifier(nn.Module):
    def __init__(self, emb_dim=512, mfcc_dim=MFCC_N*2, num_classes=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim + mfcc_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, emb, mfcc):
        x = torch.cat([emb, mfcc], dim=1)
        return self.fc(x)

# --- AUDIO PROCESSING ---
def resample_to_mono(y, orig_sr, target_sr=16000):
    # Convert to mono
    if len(y.shape) > 1:
        y = y.mean(axis=1)
    
    # Resample using librosa (better quality/anti-aliasing)
    if orig_sr != target_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
        
    return y.astype(np.float32)

def compute_log_mel(y, sr=SR):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel

def compute_mfcc_features(y, sr=SR):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])

def segment_audio(y, sr, win_sec=WINDOW_SEC, hop_sec=HOP_SEC):
    win = int(win_sec * sr)
    hop = int(hop_sec * sr)
    segments = []
    t = 0.0
    i = 0
    while i + win <= len(y):
        seg = y[i:i+win]
        segments.append((t, t + win_sec, seg))
        i += hop
        t += hop_sec
    if i < len(y):
        seg = y[i:]
        pad = win - len(seg)
        if pad > 0:
            seg = np.pad(seg, (0, pad))
        segments.append((t, t + win_sec, seg))
    return segments

# --- INFERENCE HELPERS ---
def prepare_features_for_cry_type(y_seg, cry_type_mel_model):
    log_mel = compute_log_mel(y_seg)
    mel_tensor = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    mfcc = compute_mfcc_features(y_seg)
    mfcc_tensor = torch.tensor(mfcc).unsqueeze(0).float().to(DEVICE)
    
    # Use the passed model
    emb = cry_type_mel_model(mel_tensor) 
    return emb, mfcc_tensor

@torch.no_grad()
def infer_status(y_seg, cry_status_model):
    log_mel = compute_log_mel(y_seg)
    mel_tensor = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    # Use the passed model
    prob_cry = cry_status_model(mel_tensor)
    prob_cry = float(prob_cry)
    
    label = "crying" if prob_cry >= CRY_THRESHOLD else "no_baby"
    return label, prob_cry

@torch.no_grad()
def infer_cry_type(y_seg, cry_type_mel_model, cry_type_fusion_model):
    emb, mfcc = prepare_features_for_cry_type(y_seg, cry_type_mel_model)
    logits = cry_type_fusion_model(emb, mfcc)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return CRY_TYPE_LABELS[idx], probs

def build_timeline(audio_path, cry_status_model, cry_type_mel_model, cry_type_fusion_model):
    y, orig_sr = sf.read(audio_path)
    y = resample_to_mono(y, orig_sr)
    segments = segment_audio(y, SR)
    
    timeline = []
    for t0, t1, seg in segments:
        # Pass models down
        status_label, prob_cry = infer_status(seg, cry_status_model)
        
        if status_label == "crying" and prob_cry >= CRY_THRESHOLD:
            final_type, _ = infer_cry_type(seg, cry_type_mel_model, cry_type_fusion_model)
            final_label = f"Cry: {final_type}"
        else:
            final_label = "Normal" # Shortened for video display
            
        timeline.append({
            "start": t0,
            "end": t1,
            "final": final_label
        })
    return timeline

def label_at_time(t, timeline):
    for seg in timeline:
        if seg["start"] <= t < seg["end"]:
            return seg["final"]
    return "Normal"

# --- MAIN PROCESS ---
def process_video(video_in, video_out, yolo_model, cry_status_model, cry_type_mel_model, cry_type_fusion_model):
    print(f"Processing {video_in}...")
    
    # 1. Extract Audio
    temp_audio = "tmp_process.wav"
    try:
        clip = mp.VideoFileClip(video_in)
        clip.audio.write_audiofile(temp_audio, logger=None)
    except Exception as e:
        print(f"Error loading video/audio: {e}")
        return

    # 2. Build Timeline (Pass all models)
    timeline = build_timeline(temp_audio, cry_status_model, cry_type_mel_model, cry_type_fusion_model)

    # 3. Process Video Frames (Silent)
    temp_video_silent = "tmp_silent.mp4"
    cap = cv2.VideoCapture(video_in)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(
        temp_video_silent,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    frame_idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break
            
        t = frame_idx / fps
        label = label_at_time(t, timeline)
        
        # YOLO Detection
        results = yolo_model(img, verbose=False)
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Dynamic Color: Red for crying, Green for normal
                color = (0, 0, 255) if "Cry" in label else (0, 255, 0)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, max(20, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            
        out.write(img)
        frame_idx += 1
        
    cap.release()
    out.release()
    
    # 4. Merge Audio and Save to Final Output
    final_clip = mp.VideoFileClip(temp_video_silent)
    original_audio = mp.AudioFileClip(temp_audio)
    
    final_clip = final_clip.set_audio(original_audio)
    final_clip.write_videofile(video_out, codec="libx264", audio_codec="aac", logger=None)
    
    # Cleanup temps
    final_clip.close()
    original_audio.close()
    if os.path.exists(temp_audio): os.remove(temp_audio)
    if os.path.exists(temp_video_silent): os.remove(temp_video_silent)