# seg
Segmentation with Deeplab v3 +


## install below before installing requirents.txt

### 1. Create venv
python -m venv odvenv

### 2. Activate venv
Windows:
odvenv\Scripts\activate

### 3. Install CUDA-enabled PyTorch

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

### 4. Install remaining dependencies

pip install -r requirements.txt

(<removed torch torchvision from requirements.txt>)


## Weights Download 

BACKBONE_URLS = {
    # Standard ResNet (PyTorch official)
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    
    # ResNetV1b (from encoding repo)
    "resnet50_v1b": "https://hangzh.s3.amazonaws.com/encoding/models/resnet50-25c4b594.pth",
    "resnet101_v1b": "https://hangzh.s3.amazonaws.com/encoding/models/resnet101-2a57e44d.pth",
    
    # MobileNetV2 (PyTorch official)
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}