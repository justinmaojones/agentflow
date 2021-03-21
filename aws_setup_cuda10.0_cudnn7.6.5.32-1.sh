# Add NVIDIA package repositories
# Add HTTPS support for apt-key
sudo apt-get install gnupg-curl
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /"
sudo apt-get update

# Install development and runtime libraries (~4GB)
sudo apt-get install --no-install-recommends \
    cuda-10-0 \
    libcudnn7=7.6.5.32-1+cuda10.0 \
    libcudnn7-dev=7.6.5.32-1+cuda10.0
