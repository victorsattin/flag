# Installation Requirements

## Python Version
- Python 3.11

## Creating Environment

Create an environment called `flag` with Python 3.11:

```bash
conda create --name flag python=3.11
```

Activate the environment:

```bash
conda activate flag
```
## Dependencies

Install all required packages using Anaconda PowerShell or terminal:

```bash
pip install numpy==1.26.4 opencv-python==4.10.0.84 pillow==10.4.0 matplotlib==3.9.0 seaborn==0.13.2 scikit-learn==1.5.1 ultralytics==8.2.70 torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
```

Then

```bash
-c conda-forge opencv
```

## Optional (GPU Support - CUDA)

For GPU acceleration, install the CUDA-compatible PyTorch version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Dataset Structure

The dataset folder must be organized as:

```text
dataset/
├── Germany/
├── Belgium/
├── Netherlands/
├── Canada/
├── Denmark/
└── ...
```

Each subfolder must contain the images corresponding to one flag class.

## Creating the Dataset with OpenCV

To build the image dataset, place all source videos inside a folder named:

```text
videos/
```

Example:

```text
project/
├── videos/
│   ├── germany.mp4
│   ├── belgium.mp4
│   └── canada.mp4
```

Frames can be extracted from the videos using OpenCV and saved as images for the dataset.

The OpenCV script will extract frames from the video and generate images for the dataset.

### Final Dataset Structure

```text
dataset/
├── Germany/
├── Belgium/
├── Netherlands/
├── Canada/
└── Denmark/
```

Each class folder should contain the extracted images corresponding to one flag category.

## Notes

Pretrained models are downloaded automatically on first run:

- ResNet18 (`torchvision`)
- YOLOv8n (`ultralytics`)

## Compatibility Note

This project uses **NumPy 1.26.4** for compatibility with OpenCV, PyTorch and Ultralytics.

**NumPy 2.x is not recommended**, as it may cause import errors such as:

```text
ImportError: numpy.core.multiarray failed to import
```
