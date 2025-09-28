# How to Setup-NVIDIA-GPU-for-Deep-Learning Tasks

Watch this video for a complete step-by-step walkthrough. 
- **Note:** Right-click to Open any link in a New Tab.

[![Watch the video](https://img.youtube.com/vi/zhoA3k6II5I/maxresdefault.jpg)](https://youtu.be/zhoA3k6II5I)
### [Watch the Above Video on YouTube](https://youtu.be/zhoA3k6II5I)

**Continue Reading Below:** Alternatively, you can follow the 7-Steps below the video, to learn HOW TO FULLY SETUP NVIDIA GPU FOR DEEP LEARNING.

## Step #1: Install NVIDIA Video Driver

First, you need to install the latest version of *drivers* for your *nVIDIA GPUs*. 
You can download *nVIDIA GPU drivers* using the link below: 
 - **[Click Here > Download NVIDIA GPU Drivers](https://www.nvidia.com/Download/index.aspx)**
 <!-- - **<a href="https://www.nvidia.com/Download/index.aspx" target="blank">Click Here - Download NVIDIA GPU Drivers</a>**-->

## Step #2: Install Visual Studio & C++

You will need to install MS Visual Studio, with C++ libraries. 
Note that C++ libraries are not installed with Visual Studio by default, Therefore, we msut select all of the C++ options that needs to be installed.
You can download *Visual Studio Community Edition* using the link below: 
 - **[Click Here > Download Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/)**
 <!-- - **<a href="https://visualstudio.microsoft.com/vs/community/" target="_blank">Click Here - Visual Studio Community Edition</a>** -->

## Step #3: Install Anaconda/Miniconda

In order to install all deep learning packages, we will need an Anaconda installation. 
You can download *Anaconda* using the link below: 
 - **[Click Here > Download Anaconda](https://www.anaconda.com/download/success)**
 <!-- - **<a href="https://www.anaconda.com/download/success" target="blank">Click Here - Download Anaconda</a>**-->

## Step #4: Install CUDA Toolkit

 - **[Click Here > Download CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)**
 <!-- - **<a href="https://developer.nvidia.com/cuda-toolkit-archive" target="blank">Click Here - Download CUDA Toolkit</a>**-->
 When you click on the above link, you will find that there are many versions of *CUDA Toolkit*. In order to determine which version you should download for your specific *GPU*, do the following:
 - i. Open the *Command Prompt* window on your PC.
 - ii. In the *COmmand Prompt*, Type: 
 '''python
 nvidia-msi
 '''
 - Press Enter. You will see your entire *GPU configuration*, including the *'Driver Version:'* number, and *'CUDA Version:'* number displayed (see screenshot below).
 
 The *CUDA Version* number actually shows you the *maximum* CUDA version that you can install from [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive).



## Step #5: Install cuDNN

 - **[Click Here > Download cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)**
 <!-- - **<a href="https://developer.nvidia.com/rdp/cudnn-archive" target="blank">Click Here - Download cuDNN</a>**-->


## Step #6: Install PyTorch 

 - **[Click Here > Download & Install PyTorch](https://pytorch.org/get-started/locally/)**
 <!-- - **<a href="https://pytorch.org/get-started/locally/" target="blank">Click Here - Download & Install PyTorch</a>**-->




## Step #7: Testing The GPU
This is the *Final* step to confirm the your GPU is properly installed/configured. To do so, *Run* the following script to test the GPU.

```python
import torch

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
```
