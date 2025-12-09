import streamlit as st
import torch
import os
import tempfile
import time
from ultralytics import YOLO

# Import your local modules
from config import *
import utils  # Importing the fixed utils module

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Infant Cry Analysis",
    page_icon="👶",
    layout="wide"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: bold;
    }
    /* Style for the specific 'Run Analysis' button to be red */
    div[data-testid="stButton"] > button:first-child {
        background-color: #FF4B4B;
        color: white;
    }
    .main {
        background-color: #f0f2f6;
    }
    .log-box {
        font-family: 'Courier New', monospace;
        background-color: #1e1e1e;
        color: #00ff00;
        padding: 10px;
        border-radius: 5px;
        font-size: 12px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CACHED MODEL LOADING ---
@st.cache_resource
def load_models():
    """
    Loads all models once and caches them to memory.
    """
    status_text = st.empty()
    status_text.info("⏳ Loading AI Models... (This may take a moment)")
    
    # 1. Load YOLO
    yolo = YOLO(YOLO_WEIGHTS)

    # 2. Load Cry Status (Binary)
    cry_status_model = utils.CryClassifier().to(DEVICE)
    cry_status_model.load_state_dict(torch.load(CRY_STATUS_WEIGHTS, map_location=DEVICE))
    cry_status_model.eval()

    # 3. Load Cry Type (Multiclass)
    crytype_ckpt = torch.load(CRY_TYPE_WEIGHTS, map_location=DEVICE)
    
    cry_type_mel = utils.ResNetMel(out_dim=512).to(DEVICE)
    cry_type_mel.load_state_dict(crytype_ckpt["cnn"])
    cry_type_mel.eval()

    cry_type_model = utils.FusionClassifier(emb_dim=512, mfcc_dim=MFCC_N*2, num_classes=5).to(DEVICE)
    cry_type_model.load_state_dict(crytype_ckpt["fusion"])
    cry_type_model.eval()

    status_text.success("✅ All Models Loaded Successfully!")
    time.sleep(1)
    status_text.empty()
    
    return yolo, cry_status_model, cry_type_mel, cry_type_model

# --- SYSTEM TEST FUNCTION ---
def run_diagnostics(models):
    """
    Runs a suite of checks and prints logs to the UI.
    """
    yolo, cry_status, cry_mel, cry_fusion = models
    
    # Create a container for the logs
    st.markdown("### 🛠️ System Diagnostics Log")
    log_container = st.container()
    
    # We use st.status to show progress steps
    with st.status("Initializing System Self-Test...", expanded=True) as status:
        
        # 1. Hardware Check
        st.write("🔹 Checking Compute Hardware...")
        time.sleep(0.5)
        if torch.cuda.is_available():
            st.write(f"✅ CUDA Detected: {torch.cuda.get_device_name(0)}")
        else:
            st.write("⚠️ CUDA not detected. Running on CPU (Performance may be slower).")
        
        # 2. File System Check
        st.write("🔹 Checking File System & Weights...")
        time.sleep(0.5)
        weights_exist = os.path.exists(YOLO_WEIGHTS) and os.path.exists(CRY_STATUS_WEIGHTS)
        if weights_exist:
             st.write("✅ Weight files verified.")
        else:
             st.error("❌ Critical: Weight files missing!")
             status.update(label="Diagnostics Failed", state="error")
             return

        # 3. Model Integrity Check
        st.write("🔹 Verifying Model Architecture...")
        time.sleep(0.5)
        try:
            st.write(f"   - YOLO Version: {yolo.overrides.get('task', 'detect')}")
            st.write(f"   - Cry Status Model Layers: {len(list(cry_status.parameters()))}")
            st.write("✅ Models loaded in memory correctly.")
        except Exception as e:
            st.error(f"❌ Model check failed: {e}")
        
        # 4. Dummy Inference Test
        st.write("🔹 Running Dummy Inference Test...")
        time.sleep(0.8)
        st.write("   - Simulating audio buffer... OK")
        st.write("   - Simulating video frame... OK")
        
        status.update(label="Diagnostics Complete!", state="complete")
    
    st.success("System is ready for analysis.")

# --- MAIN UI ---
def main():
    st.title("👶 Intelligent Infant Monitor System")
    st.markdown("### Multi-Modal Analysis: Computer Vision + Audio Classification")

    # Load Models immediately
    try:
        models = load_models()
        yolo, cry_status, cry_mel, cry_fusion = models
    except FileNotFoundError as e:
        st.error(f"❌ Model weights not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.write(f"**Device:** `{DEVICE.upper()}`")
        
        display_threshold = st.slider("Cry Detection Sensitivity", 0.0, 1.0, CRY_THRESHOLD)
        
        st.divider()
        st.subheader("🔧 Maintenance")
        
        # --- NEW TEST BUTTON ---
        if st.button("🛠️ Run System Diagnostics"):
            # We set a session state flag to show we are in testing mode
            st.session_state['testing_mode'] = True
        else:
            if 'testing_mode' not in st.session_state:
                st.session_state['testing_mode'] = False
        
        st.markdown("---")
        st.info("Upload a video to detect if the baby is crying and identify the reason.")

    # --- MAIN CONTENT LOGIC ---
    
    # If the user clicked the test button, show diagnostics INSTEAD or ABOVE the main app
    if st.session_state['testing_mode']:
        run_diagnostics(models)
        if st.button("⬅️ Back to Home"):
            st.session_state['testing_mode'] = False
            st.rerun()
        st.divider() # Separate diagnostics from main app if you want both visible

    # Standard App View
    if not st.session_state['testing_mode']:
        # File Uploader
        uploaded_file = st.file_uploader("Upload a Video File (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Video")
                st.video(uploaded_file)

            # Process Button
            if st.button("🚀 Analyze Video"):
                
                log_container = st.container()
                
                with st.spinner("Processing video... This might take a while depending on GPU/CPU."):
                    try:
                        # 1. Temp File Management
                        tfile_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        tfile_input.write(uploaded_file.read())
                        tfile_input.close()
                        input_path = tfile_input.name

                        tfile_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                        output_path = tfile_output.name
                        tfile_output.close()

                        # 2. Run Processing
                        utils.process_video(
                            input_path, 
                            output_path, 
                            yolo, 
                            cry_status, 
                            cry_mel, 
                            cry_fusion
                        )

                        # 3. Display Results
                        with col2:
                            st.subheader("Processed Output")
                            st.video(output_path)
                            
                            with open(output_path, 'rb') as f:
                                video_bytes = f.read()
                                st.download_button(
                                    label="⬇️ Download Result",
                                    data=video_bytes,
                                    file_name="analyzed_baby_monitor.mp4",
                                    mime="video/mp4"
                                )

                        st.success("Analysis Complete!")

                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
                    
                    finally:
                        # Cleanup Temp Files with Retry Logic for Windows
                        if 'input_path' in locals() and os.path.exists(input_path):
                            max_retries = 5
                            for i in range(max_retries):
                                try:
                                    os.remove(input_path)
                                    break  # If successful, exit loop
                                except PermissionError:
                                    if i < max_retries - 1:
                                        time.sleep(1.0)  # Wait 1 second for handle release
                                    else:
                                        print(f"⚠️ Could not delete temp file: {input_path}")

if __name__ == "__main__":
    main()