# To run the training code, 
1. you can either use the following kaggle notebook: https://www.kaggle.com/code/fangirl27/hfd-end-term
2. Follow the instructions below to train it locally

# Baby Detection & Cry Classification Notebook

This folder contains the notebook `hfd-end-term.ipynb` which trains/demonstrates two main pipelines:
- YOLO face/baby detector (uses a Roboflow-exported dataset named `roboflow_dataset`)
- Infant cry detection and cry-type classification (uses an audio dataset)

This README explains the datasets required, where to place them, and how to run the notebook locally.

**Important:** The notebook expects a small set of specific folder names (relative to the `train/` folder). You can either place datasets exactly where indicated below or update the top-level `config.py` (one folder up) to point to the dataset/model file locations.

**What to download and where to put it**

- **Roboflow baby-face dataset (for YOLO training / detection)**
  - Download from your Roboflow project (or any YOLO-formatted dataset that contains `images/` and `labels/`).
  - Put the exported dataset in:
    - `train/roboflow_dataset/`
  - Required structure (example):
    - `train/roboflow_dataset/train/images/` (jpg/png files)
    - `train/roboflow_dataset/train/labels/` (YOLO .txt label files)
    - `train/roboflow_dataset/valid/...`
    - `train/roboflow_dataset/test/...`
  - The notebook writes `data.yaml` pointing to `roboflow_dataset` by default.

- **Listening Beyond The Cry (audio dataset) — classification data**
  - The notebook expects the audio dataset to be available under a `DATA_ROOT` path. Two options:
    - Put the dataset at `train/listening-beyond-the-cry/Final Dataset/` so that audio files live under `Final Dataset/Cry/` and `Final Dataset/Not-cry/` (matching the original Kaggle layout).
    - OR set `DATA_ROOT` in the top-level `config.py` (one folder up) to point to wherever you place the dataset (absolute or relative path).
  - Example final structure if placed under `train/`:
    - `train/listening-beyond-the-cry/Final Dataset/Cry/*.wav`
    - `train/listening-beyond-the-cry/Final Dataset/Not-cry/*.wav`

- **Pretrained / checkpoint files (optional but recommended)**
  - If you already have a trained cry-classifier checkpoint, place it (or name it) so the notebook can load it.
  - Recommended locations/keys:
    - `train/best_infantcry_model.pth` — the notebook will try to load `MODEL_PATH` from the top-level `config.py` or fallback to a local filename. Update `MODEL_PATH` or `CRY_TYPE_WEIGHTS` in `config.py` to point to this file.
    - `train/notcry_cry_classifier.pth` — cry/no-cry binary classifier state dict (if used).
    - `train/yolo11n.pt` or `train/best_yolo.pt` — YOLO initial weights or your trained detector. The notebook uses the `YOLO()` call; you can set the `YOLO_WEIGHTS` value in `config.py` to point here.

**Recommended `config.py` edits (one folder up)**

Open `config.py` (the notebook appends the parent folder to `sys.path` and imports it). Add or update the following variables so the notebook finds files locally (example):

```python
# in top-level config.py (one folder up from this notebook)
DATA_ROOT = r"./train/listening-beyond-the-cry/Final Dataset"   # or absolute path
DATA_DIR = DATA_ROOT
MODEL_PATH = r"./train/best_infantcry_model.pth"
TEST_AUDIO = r"./train/example_test.wav"   # optional
INPUT_VIDEO1 = r"./train/example_video.mp4"
YOLO_WEIGHTS = r"./train/yolo11n.pt"
```

Using `config.py` keeps the notebook portable: you can point these variables to any location on your machine.

**Environment & dependencies**

- The workspace already includes a `requirements.txt` at the repo root. To install dependencies for the notebook run (PowerShell):

```powershell
python -m pip install -r ..\requirements.txt
# OR from the repository root:
python -m pip install -r requirements.txt
```

- The notebook uses `moviepy` which requires `ffmpeg` be installed and available on PATH. On Windows, install ffmpeg and add it to PATH or install `imageio[ffmpeg]`.

**Run the notebook**

1. Ensure datasets and optional checkpoint files are placed as described above (or configure `config.py`).
2. Start Jupyter Notebook / JupyterLab in the repository root, then open:
   - `train/hfd-end-term.ipynb`
3. Run the cells in order. The notebook contains a `%pip install ...` cell to install missing Python packages in the notebook environment — run that first.
4. Follow any prompts and check printed paths at the top of the notebook to ensure it finds `DATA_ROOT`, `MODEL_PATH`, `INPUT_VIDEO1`, and `YOLO_WEIGHTS`.

Example minimal steps (PowerShell):

```powershell
# from repo root
python -m pip install -r requirements.txt
jupyter lab
# then open train/hfd-end-term.ipynb and run cells
```

**Notebook usage notes / troubleshooting**

- If files are not found, the notebook prints clear errors — first check the `DATA_ROOT` and `MODEL_PATH` values printed at the top of the notebook.
- If `moviepy` fails with codec/ffmpeg errors, ensure `ffmpeg` is installed and on PATH.
- If `librosa` raises errors about `numpy` incompatibility, consider aligning `librosa` and `numpy` versions; the project `requirements.txt` can be adjusted (ask me to suggest compatible pins).
- On Windows, some paths with spaces can cause problems when launching shell commands from within notebook cells. Prefer using raw string paths in `config.py` or short paths without spaces.

**Expected outputs**

- YOLO training (if run) will produce `runs/detect/train/` outputs and an exported model `best.pt` in that folder.
- Cry classification training will save `best_infantcry_model.pth` (if you run the training fold) and a `notcry_cry_classifier.pth` for binary cry detection.
- The detection/testing cells will write videos such as `output_with_box.mp4` into the `train/` folder unless modified.

If you want, I can:
- Add example `config.py` entries to the top-level `config.py` for these dataset paths,
- Suggest exact dependency pins to avoid `librosa`/`numpy` issues,
- Or produce a small helper script to validate the dataset layout before running the notebook.

