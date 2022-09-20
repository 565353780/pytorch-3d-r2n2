conda create -n 3dr2n2 python=3.8
conda activate 3dr2n2
pip install numpy EasyDict Pillow pyyaml sklearn
pip install torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu113

