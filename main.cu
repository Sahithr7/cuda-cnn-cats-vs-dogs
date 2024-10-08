// cnn_cuda.cu

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <filesystem>
#include <string>
#include <algorithm>
#include <utility>
#include <random>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace fs = std::filesystem;

// CUDA error checking macro
#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                               \
    if(e!=cudaSuccess) {                                            \
        std::cerr << "CUDA Error " << __FILE__ << ":" << __LINE__;  \
        std::cerr << " code:" << e << ", reason: " << cudaGetErrorString(e) << std::endl; \
        exit(1);                                                    \
    }                                                               \
}

// ConvolutionalLayer class definition with CUDA
class ConvolutionalLayer {
public:
    int num_filters;
    int filter_size;
    int depth;
    float* d_filters; // Device memory for filters
    float* d_biases;  // Device memory for biases
    float* d_input;   // Device memory for input
    float* d_output;  // Device memory for output
    int input_height;
    int input_width;
    int output_height;
    int output_width;

    ConvolutionalLayer(int num_filters, int depth, int filter_size) {
        this->num_filters = num_filters;
        this->depth = depth;
        this->filter_size = filter_size;

        int filter_volume = num_filters * depth * filter_size * filter_size;
        float* h_filters = new float[filter_volume];
        float* h_biases = new float[num_filters];

        // Initialize filters and biases
        for (int i = 0; i < filter_volume; i++) {
            h_filters[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
        }
        for (int i = 0; i < num_filters; i++) {
            h_biases[i] = 0.0f;
        }

        // Allocate device memory
        cudaMalloc(&d_filters, filter_volume * sizeof(float));
        cudaMalloc(&d_biases, num_filters * sizeof(float));

        // Copy filters and biases to device
        cudaMemcpy(d_filters, h_filters, filter_volume * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_biases, h_biases, num_filters * sizeof(float), cudaMemcpyHostToDevice);

        delete[] h_filters;
        delete[] h_biases;
    }

    ~ConvolutionalLayer() {
        // Free device memory
        cudaFree(d_filters);
        cudaFree(d_biases);
    }

    // CUDA kernel for convolution forward pass
    __global__ void conv_forward_kernel(const float* __restrict__ d_input,
                                        const float* __restrict__ d_filters,
                                        const float* __restrict__ d_biases,
                                        float* d_output,
                                        int depth, int input_height, int input_width,
                                        int filter_size, int output_height, int output_width) {
        int n = blockIdx.z; // Filter index
        int i = blockIdx.y * blockDim.y + threadIdx.y; // Output height index
        int j = blockIdx.x * blockDim.x + threadIdx.x; // Output width index

        if (i < output_height && j < output_width) {
            float sum = d_biases[n];
            for (int d = 0; d < depth; d++) {
                for (int fi = 0; fi < filter_size; fi++) {
                    for (int fj = 0; fj < filter_size; fj++) {
                        int input_i = i + fi;
                        int input_j = j + fj;
                        int input_idx = d * input_height * input_width + input_i * input_width + input_j;
                        int filter_idx = n * depth * filter_size * filter_size + d * filter_size * filter_size + fi * filter_size + fj;
                        sum += d_input[input_idx] * d_filters[filter_idx];
                    }
                }
            }
            int output_idx = n * output_height * output_width + i * output_width + j;
            d_output[output_idx] = sum;
        }
    }

    // Forward propagation
    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input) {
        // Set input dimensions
        input_height = input[0].size();
        input_width = input[0][0].size();

        int input_size = depth * input_height * input_width;
        float* h_input = new float[input_size];

        // Flatten input
        int idx = 0;
        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < input_height; i++) {
                for (int j = 0; j < input_width; j++) {
                    h_input[idx++] = input[d][i][j];
                }
            }
        }

        // Allocate device memory for input and output
        cudaMalloc(&d_input, input_size * sizeof(float));

        output_height = input_height - filter_size + 1;
        output_width = input_width - filter_size + 1;
        int output_size = num_filters * output_height * output_width;
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        delete[] h_input;

        // Launch CUDA kernel for convolution
        dim3 blockDim(16, 16);
        dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x,
                     (output_height + blockDim.y - 1) / blockDim.y,
                     num_filters);

        conv_forward_kernel<<<gridDim, blockDim>>>(d_input, d_filters, d_biases, d_output,
                                                   depth, input_height, input_width,
                                                   filter_size, output_height, output_width);

        cudaCheckError();
        cudaDeviceSynchronize();

        // Copy output back to host
        int output_size_host = num_filters * output_height * output_width;
        float* h_output = new float[output_size_host];
        cudaMemcpy(h_output, d_output, output_size_host * sizeof(float), cudaMemcpyDeviceToHost);

        // Reshape output to 3D vector
        std::vector<std::vector<std::vector<float>>> output(num_filters,
            std::vector<std::vector<float>>(output_height, std::vector<float>(output_width)));

        idx = 0;
        for (int n = 0; n < num_filters; n++) {
            for (int i = 0; i < output_height; i++) {
                for (int j = 0; j < output_width; j++) {
                    output[n][i][j] = h_output[idx++];
                }
            }
        }

        delete[] h_output;

        // Free device input (keep d_output for backward pass)
        cudaFree(d_input);

        return output;
    }

    // Backward propagation (to be implemented similarly with CUDA)
    // For brevity, the backward function is not fully implemented here.
    // You would need to write CUDA kernels for the backward pass as well.
};

// ReLULayer class definition with CUDA
class ReLULayer {
public:
    float* d_input;   // Device input
    float* d_output;  // Device output
    int size;

    // CUDA kernel for ReLU forward pass
    __global__ void relu_forward_kernel(const float* d_input, float* d_output, int size) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < size) {
            d_output[idx] = fmaxf(0.0f, d_input[idx]);
        }
    }

    // Forward propagation
    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input) {
        // Flatten input
        int depth = input.size();
        int height = input[0].size();
        int width = input[0][0].size();
        size = depth * height * width;
        float* h_input = new float[size];

        int idx = 0;
        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    h_input[idx++] = input[d][i][j];
                }
            }
        }

        // Allocate device memory
        cudaMalloc(&d_input, size * sizeof(float));
        cudaMalloc(&d_output, size * sizeof(float));

        // Copy input to device
        cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch ReLU kernel
        int threads = 256;
        int blocks = (size + threads - 1) / threads;
        relu_forward_kernel<<<blocks, threads>>>(d_input, d_output, size);

        cudaCheckError();
        cudaDeviceSynchronize();

        // Copy output back to host
        float* h_output = new float[size];
        cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

        // Reshape output to 3D vector
        std::vector<std::vector<std::vector<float>>> output(depth,
            std::vector<std::vector<float>>(height, std::vector<float>(width)));

        idx = 0;
        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    output[d][i][j] = h_output[idx++];
                }
            }
        }

        delete[] h_input;
        delete[] h_output;

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);

        return output;
    }

    // Backward propagation (to be implemented similarly with CUDA)
};

// MaxPoolingLayer class definition with CUDA
class MaxPoolingLayer {
public:
    int pool_size;
    int stride;
    int depth;
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    float* d_input;
    float* d_output;
    int* d_max_indices;

    MaxPoolingLayer(int pool_size, int stride) {
        this->pool_size = pool_size;
        this->stride = stride;
    }

    // CUDA kernel for MaxPooling forward pass
    __global__ void maxpool_forward_kernel(const float* __restrict__ d_input,
                                           float* d_output,
                                           int* d_max_indices,
                                           int depth, int input_height, int input_width,
                                           int pool_size, int stride,
                                           int output_height, int output_width) {
        int d = blockIdx.z;
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < output_height && j < output_width) {
            int input_base = d * input_height * input_width;
            int output_base = d * output_height * output_width;
            int max_idx = -1;
            float max_val = -INFINITY;

            for (int pi = 0; pi < pool_size; pi++) {
                for (int pj = 0; pj < pool_size; pj++) {
                    int ni = i * stride + pi;
                    int nj = j * stride + pj;
                    int idx = input_base + ni * input_width + nj;
                    float val = d_input[idx];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = idx;
                    }
                }
            }
            int out_idx = output_base + i * output_width + j;
            d_output[out_idx] = max_val;
            d_max_indices[out_idx] = max_idx;
        }
    }

    // Forward propagation
    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input) {
        depth = input.size();
        input_height = input[0].size();
        input_width = input[0][0].size();

        output_height = (input_height - pool_size) / stride + 1;
        output_width = (input_width - pool_size) / stride + 1;

        int input_size = depth * input_height * input_width;
        int output_size = depth * output_height * output_width;

        float* h_input = new float[input_size];

        // Flatten input
        int idx = 0;
        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < input_height; i++) {
                for (int j = 0; j < input_width; j++) {
                    h_input[idx++] = input[d][i][j];
                }
            }
        }

        // Allocate device memory
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));
        cudaMalloc(&d_max_indices, output_size * sizeof(int));

        // Copy input to device
        cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch MaxPooling kernel
        dim3 blockDim(16, 16);
        dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x,
                     (output_height + blockDim.y - 1) / blockDim.y,
                     depth);

        maxpool_forward_kernel<<<gridDim, blockDim>>>(d_input, d_output, d_max_indices,
                                                      depth, input_height, input_width,
                                                      pool_size, stride,
                                                      output_height, output_width);

        cudaCheckError();
        cudaDeviceSynchronize();

        // Copy output back to host
        float* h_output = new float[output_size];
        cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        // Reshape output to 3D vector
        std::vector<std::vector<std::vector<float>>> output(depth,
            std::vector<std::vector<float>>(output_height, std::vector<float>(output_width)));

        idx = 0;
        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < output_height; i++) {
                for (int j = 0; j < output_width; j++) {
                    output[d][i][j] = h_output[idx++];
                }
            }
        }

        delete[] h_input;
        delete[] h_output;

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_max_indices);

        return output;
    }

    // Backward propagation (to be implemented similarly with CUDA)
};

// FullyConnectedLayer class definition with CUDA
class FullyConnectedLayer {
public:
    int input_size;
    int output_size;
    float* d_weights; // Device weights
    float* d_biases;  // Device biases
    float* d_input;
    float* d_output;

    FullyConnectedLayer(int input_size, int output_size) {
        this->input_size = input_size;
        this->output_size = output_size;

        int weights_size = output_size * input_size;
        float* h_weights = new float[weights_size];
        float* h_biases = new float[output_size];

        // Initialize weights and biases
        for (int i = 0; i < weights_size; i++) {
            h_weights[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
        }
        for (int i = 0; i < output_size; i++) {
            h_biases[i] = 0.0f;
        }

        // Allocate device memory
        cudaMalloc(&d_weights, weights_size * sizeof(float));
        cudaMalloc(&d_biases, output_size * sizeof(float));

        // Copy weights and biases to device
        cudaMemcpy(d_weights, h_weights, weights_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_biases, h_biases, output_size * sizeof(float), cudaMemcpyHostToDevice);

        delete[] h_weights;
        delete[] h_biases;
    }

    ~FullyConnectedLayer() {
        // Free device memory
        cudaFree(d_weights);
        cudaFree(d_biases);
    }

    // CUDA kernel for fully connected forward pass
    __global__ void fc_forward_kernel(const float* __restrict__ d_input,
                                      const float* __restrict__ d_weights,
                                      const float* __restrict__ d_biases,
                                      float* d_output,
                                      int input_size, int output_size) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < output_size) {
            float sum = d_biases[i];
            for (int j = 0; j < input_size; j++) {
                sum += d_weights[i * input_size + j] * d_input[j];
            }
            d_output[i] = sum;
        }
    }

    // Forward propagation
    std::vector<float> forward(const std::vector<float>& input) {
        // Allocate device memory for input and output
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        // Copy input to device
        cudaMemcpy(d_input, input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        int threads = 256;
        int blocks = (output_size + threads - 1) / threads;
        fc_forward_kernel<<<blocks, threads>>>(d_input, d_weights, d_biases, d_output, input_size, output_size);

        cudaCheckError();
        cudaDeviceSynchronize();

        // Copy output back to host
        std::vector<float> output(output_size);
        cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_input);
        cudaFree(d_output);

        return output;
    }

    // Backward propagation (to be implemented similarly with CUDA)
};

// Sigmoid function
__device__ __host__ float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Derivative of sigmoid function
__device__ __host__ float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

// Function to load images and labels
std::vector<std::pair<std::vector<std::vector<std::vector<float>>>, int>> load_data(const std::string& data_dir, int image_height, int image_width) {
    std::vector<std::pair<std::vector<std::vector<std::vector<float>>>, int>> dataset; // pair of image and label

    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (fs::is_directory(entry)) {
            std::string label_str = entry.path().filename().string();
            int label = (label_str == "cats") ? 0 : 1;
            for (const auto& image_file : fs::directory_iterator(entry.path())) {
                std::string image_path = image_file.path().string();
                int width, height, channels;
                unsigned char* img = stbi_load(image_path.c_str(), &width, &height, &channels, 0);
                if (img == nullptr) {
                    std::cerr << "Failed to load image: " << image_path << std::endl;
                    continue;
                }

                // Resize image to target dimensions
                std::vector<std::vector<std::vector<float>>> image_data(1, std::vector<std::vector<float>>(image_height, std::vector<float>(image_width, 0.0f)));

                if (channels >= 3) {
                    // Convert to grayscale and resize
                    for (int i = 0; i < image_height; i++) {
                        for (int j = 0; j < image_width; j++) {
                            int img_i = i * height / image_height;
                            int img_j = j * width / image_width;
                            int idx = (img_i * width + img_j) * channels;
                            unsigned char r = img[idx];
                            unsigned char g = img[idx + 1];
                            unsigned char b = img[idx + 2];
                            float gray = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
                            image_data[0][i][j] = gray;
                        }
                    }
                } else if (channels == 1) {
                    // Grayscale image, resize
                    for (int i = 0; i < image_height; i++) {
                        for (int j = 0; j < image_width; j++) {
                            int img_i = i * height / image_height;
                            int img_j = j * width / image_width;
                            int idx = img_i * width + img_j;
                            unsigned char v = img[idx];
                            image_data[0][i][j] = v / 255.0f;
                        }
                    }
                }

                stbi_image_free(img);
                dataset.push_back(std::make_pair(image_data, label));
            }
        }
    }
    return dataset;
}

int main() {
    srand(time(0));

    // Set image dimensions
    int image_height = 128;
    int image_width = 128;
    int image_depth = 1; // Grayscale

    // Load training data
    std::string train_data_dir = "path/to/train_data";
    auto train_dataset = load_data(train_data_dir, image_height, image_width);

    // Load test data
    std::string test_data_dir = "path/to/test_data";
    auto test_dataset = load_data(test_data_dir, image_height, image_width);

    // Build the network
    ConvolutionalLayer conv1(8, image_depth, 3); // 8 filters, filter size 3x3
    ReLULayer relu1;
    MaxPoolingLayer pool1(2, 2); // Pool size 2x2, stride 2

    // Compute output size after conv and pool layers
    int conv1_out_height = image_height - 3 + 1; // (input_height - filter_size + 1)
    int conv1_out_width = image_width - 3 + 1;

    int pool1_out_height = (conv1_out_height - 2) / 2 + 1;
    int pool1_out_width = (conv1_out_width - 2) / 2 + 1;

    int fc_input_size = 8 * pool1_out_height * pool1_out_width;

    FullyConnectedLayer fc1(fc_input_size, 64); // Hidden layer with 64 neurons
    FullyConnectedLayer fc2(64, 1); // Output layer (binary classification)

    // Training loop
    int num_epochs = 10;
    float learning_rate = 0.001f;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << std::endl;

        // Shuffle the training data
        auto rng = std::default_random_engine{};
        std::shuffle(std::begin(train_dataset), std::end(train_dataset), rng);

        float total_loss = 0.0f;
        int correct_predictions = 0;

        for (auto& data_pair : train_dataset) {
            auto& image = data_pair.first;
            int label = data_pair.second;

            // Forward pass
            auto output_conv1 = conv1.forward(image);
            auto output_relu1 = relu1.forward(output_conv1);
            auto output_pool1 = pool1.forward(output_relu1);

            // Flatten output of pool1
            std::vector<float> flattened_input;
            for (int d = 0; d < output_pool1.size(); d++) {
                for (int i = 0; i < output_pool1[0].size(); i++) {
                    for (int j = 0; j < output_pool1[0][0].size(); j++) {
                        flattened_input.push_back(output_pool1[d][i][j]);
                    }
                }
            }

            // Fully connected layers
            auto output_fc1 = fc1.forward(flattened_input);
            // Apply ReLU activation
            std::vector<float> relu_fc1(output_fc1.size(), 0.0f);
            for (int i = 0; i < output_fc1.size(); i++) {
                relu_fc1[i] = std::max(0.0f, output_fc1[i]);
            }
            auto output_fc2 = fc2.forward(relu_fc1);

            // Apply sigmoid activation to the output
            float output = sigmoid(output_fc2[0]);

            // Compute loss
            float y = label; // y = 0 or 1
            float loss = - (y * log(output + 1e-7f) + (1 - y) * log(1 - output + 1e-7f));

            // Compute gradient of loss w.r.t output
            float d_loss_output = - (y / (output + 1e-7f)) + ((1 - y) / (1 - output + 1e-7f));

            // Backward pass (not implemented with CUDA here)
            // You would need to implement backward methods for each layer using CUDA

            // Accumulate loss and accuracy
            total_loss += loss;
            if ((output >= 0.5f && label == 1) || (output < 0.5f && label == 0)) {
                correct_predictions++;
            }
        }

        // Compute average loss and accuracy
        float avg_loss = total_loss / train_dataset.size();
        float accuracy = (float)correct_predictions / train_dataset.size();
        std::cout << "Training Loss: " << avg_loss << ", Accuracy: " << accuracy * 100.0f << "%" << std::endl;

        // Evaluate on test data
        int test_correct = 0;
        for (auto& data_pair : test_dataset) {
            auto& image = data_pair.first;
            int label = data_pair.second;

            // Forward pass
            auto output_conv1 = conv1.forward(image);
            auto output_relu1 = relu1.forward(output_conv1);
            auto output_pool1 = pool1.forward(output_relu1);

            // Flatten output of pool1
            std::vector<float> flattened_input;
            for (int d = 0; d < output_pool1.size(); d++) {
                for (int i = 0; i < output_pool1[0].size(); i++) {
                    for (int j = 0; j < output_pool1[0][0].size(); j++) {
                        flattened_input.push_back(output_pool1[d][i][j]);
                    }
                }
            }

            // Fully connected layers
            auto output_fc1 = fc1.forward(flattened_input);
            // Apply ReLU activation
            std::vector<float> relu_fc1(output_fc1.size(), 0.0f);
            for (int i = 0; i < output_fc1.size(); i++) {
                relu_fc1[i] = std::max(0.0f, output_fc1[i]);
            }
            auto output_fc2 = fc2.forward(relu_fc1);

            // Apply sigmoid activation to the output
            float output = sigmoid(output_fc2[0]);

            // Prediction
            if ((output >= 0.5f && label == 1) || (output < 0.5f && label == 0)) {
                test_correct++;
            }
        }
        float test_accuracy = (float)test_correct / test_dataset.size();
        std::cout << "Test Accuracy: " << test_accuracy * 100.0f << "%" << std::endl;
    }

    return 0;
}
