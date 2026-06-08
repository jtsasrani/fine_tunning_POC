#!/bin/bash
set -e

echo "=== DWP CMG EC2 Setup ==="

# 1. Activate virtual environment
source ~/unsloth_env/bin/activate

# 2. Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# 3. Install PyTorch with CUDA 12.4 support
echo "Installing PyTorch with CUDA 12.4..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install Unsloth and key dependencies
echo "Installing Unsloth..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# 5. Install other required packages
echo "Installing auxiliary packages..."
pip install transformers datasets pymupdf sentence-transformers faiss-cpu rank_bm25 fastapi uvicorn flask requests boto3 scikit-learn rouge-score bert-score

# 6. Verify GPU and CUDA integration
python -c "import torch; print('PyTorch CUDA available:', torch.cuda.is_available())"
python -c "import unsloth; print('Unsloth import success!')"

echo "=== EC2 Setup Complete ==="
