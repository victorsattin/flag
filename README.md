## Installation Requirements

### Python Version
- Python 3.10+

### Dependencies

Install all required packages on Anaconda PowerShell:
pip install numpy opencv-python pillow matplotlib seaborn scikit-learn ultralytics torch torchvision torchaudio

### Optional (GPU Support - CUDA)
For GPU acceleration, install the CUDA-compatible PyTorch version:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

### Notes
Pretrained models are downloaded automatically on first run:
- ResNet18 (`torchvision`)
- YOLOv8n (`ultralytics`)
