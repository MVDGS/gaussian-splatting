# download cuda toolkit 11.6 for nvcc
# !wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_510.47.03_linux.run
# chmod +x cuda_11.6.2_510.47.03_linux.run
# sudo sh cuda_11.6.2_510.47.03_linux.run --silent --toolkit

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 \
  -f https://download.pytorch.org/whl/torch_stable.html

pip install tqdm
pip install opencv-python joblib
pip install ./submodules/diff-gaussian-rasterization ./submodules/simple-knn ./submodules/fused-ssim

# download colmap 
conda install -c conda-forge colmap

