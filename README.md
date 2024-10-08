# Handcrafted CUDA-Accelerated CNN for Cats vs. Dogs Classification

This project implements a Convolutional Neural Network (CNN) from scratch in C++ and CUDA for classifying images of cats and dogs. The neural network is hard-coded without the use of any deep learning libraries. All layers and computations are manually implemented to provide a foundational understanding of CNNs and CUDA optimization.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Download the Dataset](#2-download-the-dataset)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Adjust Paths in the Code](#4-adjust-paths-in-the-code)
- [Compiling and Running the Code](#compiling-and-running-the-code)
  - [On Windows](#on-windows)
  - [On Linux](#on-linux)
- [Project Details](#project-details)
  - [1. Data Loading](#1-data-loading)
  - [2. Network Architecture](#2-network-architecture)
  - [3. CUDA Optimization](#3-cuda-optimization)
  - [4. Training Loop](#4-training-loop)
- [Notes and Considerations](#notes-and-considerations)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Project Overview
This project aims to classify images of cats and dogs using a CNN that is entirely implemented from scratch in C++ and optimized with CUDA for GPU acceleration. The implementation includes:

### Custom CNN Layers:
- **Convolutional Layer**
- **ReLU Activation Layer**
- **Max-Pooling Layer**
- **Fully Connected Layer**

### CUDA Kernels:
- Forward and backward passes are accelerated using custom CUDA kernels.

### Image Loading:
- Uses `stb_image.h` for image loading and preprocessing.

### Dataset:
- Utilizes the Kaggle Dogs vs. Cats dataset.

### No External Deep Learning Libraries:
- All neural network components are hard-coded without using libraries like TensorFlow, PyTorch, or cuDNN.

## Directory Structure
Your project directory should be organized as follows:

```
project_directory/
├── main.cu               # The main CUDA source code file
├── stb_image.h           # Header file for image loading
├── README.md             # This readme file
├── train/                # Training dataset directory
│   ├── cats/             # Directory containing cat images
│   └── dogs/             # Directory containing dog images
└── test/                 # Testing dataset directory
    ├── cats/             # Directory containing cat images
    └── dogs/             # Directory containing dog images
```

## Features
- **Hard-Coded Neural Network**:
  - All neural network layers and operations are manually implemented.
  - Provides an educational insight into the inner workings of CNNs.
- **CUDA Acceleration**:
  - Computationally intensive operations are optimized using CUDA.
  - Custom CUDA kernels for forward and backward passes.
- **No High-Level Libraries**:
  - Only standard libraries and CUDA are used.
- **Image Preprocessing**:
  - Images are loaded and preprocessed using `stb_image.h`.
  - Supports grayscale conversion and resizing.

## Requirements
- NVIDIA GPU with CUDA capability.
- CUDA Toolkit installed on your system.
- Microsoft Visual C++ Build Tools (for Windows users).
- C++17 Compiler with support for the C++17 standard.
- `stb_image.h` header file for image loading.
- Dataset: Kaggle Dogs vs. Cats dataset.
- No External Deep Learning Libraries: All code is written from scratch without libraries like TensorFlow or PyTorch.

## Setup Instructions
### 1. Clone the Repository
Navigate to your desired directory and clone the repository:

```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository
```

Alternatively, create a new directory and place the `main.cu` and `stb_image.h` files inside.

### 2. Download the Dataset
#### a. Kaggle Dogs vs. Cats Dataset
- **Download Link**: [Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Instructions**:
  - Sign up for a Kaggle account if you don't have one.
  - Download the dataset files (`train.zip` and `test1.zip`).
  - Extract the zip files.

#### b. Organize the Dataset
- **Training Data**:
  - Create a `train` directory in your project directory.
  - Inside `train`, create two subdirectories: `cats` and `dogs`.
  - Move cat images (e.g., `cat.0.jpg`, `cat.1.jpg`, ...) into `train/cats`.
  - Move dog images (e.g., `dog.0.jpg`, `dog.1.jpg`, ...) into `train/dogs`.
- **Testing Data**:
  - Create a `test` directory in your project directory.
  - Inside `test`, create two subdirectories: `cats` and `dogs`.
  - Split some images from the training set or use the provided test set (label the images accordingly) and place them into `test/cats` and `test/dogs`.

### 3. Install Dependencies
#### a. Install CUDA Toolkit
- **Download Link**: [CUDA Toolkit Download](https://developer.nvidia.com/cuda-downloads)
- **Instructions**:
  - Download and install the appropriate version for your operating system and GPU.

#### b. Install Microsoft Visual C++ Build Tools (Windows Only)
- **Download Link**: [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- **Instructions**:
  - Download and install the Build Tools, selecting the "Desktop development with C++" workload.

#### c. Obtain `stb_image.h`
- **Download Link**: [stb_image.h](https://github.com/nothings/stb/blob/master/stb_image.h)
- **Instructions**:
  - Place `stb_image.h` in your project directory alongside `main.cu`.

### 4. Adjust Paths in the Code
In `main.cu`, update the dataset paths:

```cpp
// Load training data
std::string train_data_dir = "train";
auto train_dataset = load_data(train_data_dir, image_height, image_width);

// Load test data
std::string test_data_dir = "test";
auto test_dataset = load_data(test_data_dir, image_height, image_width);
```

## Compiling and Running the Code
### On Windows
1. **Open Developer Command Prompt**:
   - Start Menu → Visual Studio → "x64 Native Tools Command Prompt for VS 2019" (or your installed version).

2. **Navigate to Project Directory**:

   ```cmd
   cd path\to\your\project_directory
   ```

3. **Compile the Code**:

   ```cmd
   nvcc -std=c++17 -o main main.cu
   ```

4. **Run the Executable**:

   ```cmd
   main.exe
   ```
   

## Project Details
### 1. Data Loading
- **Function**: `load_data()`
- **Description**:
  - Loads images from the dataset directories.
  - Converts images to grayscale.
  - Resizes images to the specified dimensions.
  - Normalizes pixel values to the range [0, 1].
- **Implementation**:
  - Uses `stb_image.h` for image loading.
  - No external image processing libraries are used.

### 2. Network Architecture
- **Convolutional Layer**: `ConvolutionalLayer conv1(8, image_depth, 3);`
  - 8 filters of size 3x3.
  - Implemented from scratch with CUDA optimization.
- **ReLU Activation Layer**: `ReLULayer relu1;`
  - Applies the ReLU activation function.
  - Hard-coded without using activation function libraries.
- **Max-Pooling Layer**: `MaxPoolingLayer pool1(2, 2);`
  - Pool size 2x2 with stride 2.
  - Custom implementation using CUDA.
- **Fully Connected Layers**:
  - `FullyConnectedLayer fc1(fc_input_size, 64);`
    - Hidden layer with 64 neurons.
    - Manually implemented matrix operations.
  - `FullyConnectedLayer fc2(64, 1);`
    - Output layer for binary classification.
- **Activation Functions**:
  - Sigmoid Function: Implemented manually for output layer.
- **No External Libraries**: All layers and activations are hard-coded.

### 3. CUDA Optimization
- **Custom CUDA Kernels**:
  - Written for forward and backward passes of each layer.
  - Optimizes computations by leveraging GPU parallelism.
- **Memory Management**:
  - Device memory allocation and deallocation are manually handled.
  - Ensures efficient use of GPU resources.
- **No Use of cuDNN or cuBLAS**:
  - All CUDA operations are implemented without high-level CUDA libraries.
  - Provides a deeper understanding of CUDA programming.

### 4. Training Loop
- **Epochs**: Set the number of training epochs (e.g., `int num_epochs = 10;`).
- **Learning Rate**: Define the learning rate for gradient descent (e.g., `float learning_rate = 0.001f;`).
- **Process**:
  - **Data Shuffling**: Training data is shuffled each epoch.
  - **Forward Pass**: Data passes through the network layers.
  - **Loss Computation**: Binary cross-entropy loss is calculated.
  - **Backward Pass**:
    - Gradients are computed manually.
    - CUDA kernels are used to accelerate gradient computations.
  - **Weight Updates**: Weights and biases are updated using the computed gradients.
- **Evaluation**:
  - Training loss and accuracy are printed each epoch.
  - Optionally, evaluate on test data each epoch.

## Notes and Considerations
- **Implementation Complexity**:
  - Manually implementing all aspects of a CNN is complex.
  - This project is intended for educational purposes to understand the fundamentals.
- **Image Dimensions**:
  - Adjust `image_height` and `image_width` if you encounter memory issues.
- **Dataset Size**:
  - Ensure you have enough images in both train and test directories for meaningful results.
- **GPU Memory**:
  - Be mindful of your GPU's memory capacity.
- **Error Handling**:
  - Use `cudaCheckError()` after CUDA API calls to catch errors.
- **C++ Standard**:
  - Ensure your compiler supports C++17 or later.
- **Performance**:
  - Since high-level optimizations and libraries are not used, performance may not match that of frameworks like TensorFlow or PyTorch.
- **Learning Opportunity**:
  - This project provides a hands-on experience in implementing neural networks and CUDA programming.

## Acknowledgments
- **stb_image Library**: Sean Barrett for the `stb_image.h` library used for image loading.
- **Dataset**: Kaggle for providing the Dogs vs. Cats dataset.
- **CUDA Toolkit**: NVIDIA for the CUDA Toolkit and documentation.
- **Inspiration**: The desire to understand neural networks at a deeper level by implementing them from scratch.


Feel free to contribute to this project or raise issues if you encounter any problems. This project is a learning endeavor to explore the fundamentals of CNNs and CUDA programming without relying on external libraries. Happy coding!

