# How to Setup-NVIDIA-GPU-for-Deep-Learning Tasks

Watch the video below or follow the 7-steps below the video, to learn HOW TO FULLY SETUP NVIDIA GPU FOR DEEP LEARNING.

[![Watch the video](https://img.youtube.com/vi/zhoA3k6II5I/maxresdefault.jpg)](https://youtu.be/zhoA3k6II5I)
### [Watch this video on YouTube](https://youtu.be/zhoA3k6II5I)

## Step #1: Install NVIDIA Video Driver

First, you need to install the latest version of *drivers* for your *nVIDIA GPUs*. 
You can download *nVIDIA GPU drivers* using the link below: 
 - **[Click Here - Download NVIDIA GPU Drivers](https://www.nvidia.com/Download/index.aspx)**

## Step #2: Install Visual Studio & C++

You will need to install MS Visual Studio, with C++ libraries. 
Note that C++ libraries are not installed with Visual Studio by default, Therefore, we msut select all of the C++ options that needs to be installed.
You can download *Visual Studio Community Edition* using the link below: 
 - **[Click Here - Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/)**

## Step #3: Install Anaconda/Miniconda

In order to install all deep learning packages, we will need an Anaconda installation. 
You can download *Anaconda* using the link below: 
 - **[Click Here - Download Anaconda](https://www.anaconda.com/download/success)**

## Step #4: Install CUDA Toolkit

 - **[Click Here - Download CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)**

## Step #5: Install cuDNN

 - **[Click Here - Download cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)**


## Step #6: Install PyTorch 

 - **[Click Here - Download & Install PyTorch](https://pytorch.org/get-started/locally/)**




## Step #7: Testing The GPU
This is the *Final* step to confirm the your GPU is properly installed/configured. To do so, *Run* the following script to test the GPU.

```python
import torch

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
```
