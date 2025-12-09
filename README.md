
# 👶 Intelligent Infant Monitor System

An AI-powered system designed to monitor infants using multi-modal analysis. This project combines **Computer Vision (YOLO)** and **Audio Analysis (ResNet + Fusion)** to detect if a baby is present, recognize if they are crying, and classify the specific type of cry (e.g., Hungry, Pain, Tired).

## 📋 Table of Contents

  - [Prerequisites & CUDA Setup](https://www.google.com/search?q=%23-prerequisites--cuda-setup)
  - [Installation](https://www.google.com/search?q=%23-installation)
  - [Running the Project](https://www.google.com/search?q=%23-running-the-project)
      - [Option 1: Streamlit UI (Recommended)](https://www.google.com/search?q=%23option-1-streamlit-ui-recommended)
      - [Option 2: Headless Script](https://www.google.com/search?q=%23option-2-headless-script)
  - [Training Custom Models](https://www.google.com/search?q=%23-training-custom-models)

-----

## 🔌 Prerequisites & CUDA Setup

This project uses **PyTorch** and **Ultralytics**. For the best performance, you should run this on a machine with an NVIDIA GPU.

**⚠️ Important: Check your GPU Driver**
Before installing dependencies, you must verify which version of CUDA your driver supports.

1.  Open your terminal or command prompt.
2.  Run the following command:
    ```bash
    nvidia-smi
    ```
3.  Look for the **CUDA Version** in the top right corner of the table.

**Adjusting `requirements.txt`:**
The default `requirements.txt` is set for general compatibility. However, if you need a specific CUDA version (e.g., CUDA 11.8 or 12.1) to match your hardware:

1.  Open `requirements.txt`.
2.  Add the specific PyTorch index URL at the top of the file. For example, for CUDA 11.8:
    ```text
    --extra-index-url https://download.pytorch.org/whl/cu118
    ```
3.  Save the file before proceeding to installation.

-----

## 🛠 Installation

Follow these steps to set up the development environment.

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/infant-monitor-system.git
cd infant-monitor-system
```

### 2\. Create a Virtual Environment

It is highly recommended to use a virtual environment to avoid conflicts with other projects.

**For Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

Once the virtual environment is active, install the required packages:

```bash
pip install -r requirements.txt
```

-----

## 🚀 Running the Project

You can use this system in two ways: via a visual web interface or as a standalone processing script.

### Option 1: Streamlit UI (Recommended)

This launches a user-friendly web interface where you can upload videos, view the processing progress, and see the input/output side-by-side.

```bash
streamlit run app.py
```

*The app will automatically open in your default web browser.*
(Note: please scroll down to find the analyse video button)

### Option 2: Headless Script

If you want to process specific files defined in your configuration without a UI:

1.  Open `config.py`.
2.  Edit the `INPUT_VIDEO` and `OUTPUT_VIDEO` paths to point to your files.
3.  Run the main script:
    ```bash
    python main.py
    ```

*The script will process the videos and save them to the output paths defined in the config.*

-----

## 🧠 Training Custom Models

If you wish to retrain the Cry Classification models or fine-tune the YOLO detection:

1.  Navigate to the `train/` directory.
2.  Please read the specific **README** inside that folder.
3.  The folder contains notebooks and scripts for:
      * Data preprocessing (Audio MEL/MFCC extraction).
      * Training the ResNet classifiers.
      * Fine-tuning YOLO on infant datasets.

-----

## 📁 Project Structure

  * `app.py`: The Streamlit web application entry point.
  * `main.py`: The standalone script for batch processing videos.
  * `config.py`: Central configuration for file paths, model hyperparameters, and thresholds.
  * `utils.py`: Contains core logic for Model classes, Audio processing, and Inference.
  * `requirements.txt`: List of Python dependencies.
  * `final_yolo_weights_hfd/`: Directory containing trained model weights.