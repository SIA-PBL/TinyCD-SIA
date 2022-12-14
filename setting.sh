export TORCH_CUDA_ARCH_LIST=8.0
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install albumentations split-image
