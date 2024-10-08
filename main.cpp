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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace fs = std::filesystem;

// ConvolutionalLayer class definition
class ConvolutionalLayer {
public:
    int num_filters;
    int filter_size;
    int depth;
    std::vector<std::vector<std::vector<std::vector<float>>>> filters;
    std::vector<float> biases;
    std::vector<std::vector<std::vector<float>>> input_data;

    ConvolutionalLayer(int num_filters, int depth, int filter_size) {
        this->num_filters = num_filters;
        this->depth = depth;
        this->filter_size = filter_size;

        // Initialize filters and biases
        filters.resize(num_filters, std::vector<std::vector<std::vector<float>>>(depth, std::vector<std::vector<float>>(filter_size, std::vector<float>(filter_size))));
        biases.resize(num_filters, 0.0f);

        for (int n = 0; n < num_filters; n++) {
            for (int d = 0; d < depth; d++) {
                for (int i = 0; i < filter_size; i++) {
                    for (int j = 0; j < filter_size; j++) {
                        filters[n][d][i][j] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
                    }
                }
            }
        }
    }

    // Forward propagation
    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input) {
        input_data = input;
        int depth = input.size();
        int height = input[0].size();
        int width = input[0][0].size();

        int out_height = height - filter_size + 1;
        int out_width = width - filter_size + 1;

        std::vector<std::vector<std::vector<float>>> output(num_filters, std::vector<std::vector<float>>(out_height, std::vector<float>(out_width, 0.0f)));

        for (int n = 0; n < num_filters; n++) {
            for (int i = 0; i < out_height; i++) {
                for (int j = 0; j < out_width; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < depth; d++) {
                        for (int fi = 0; fi < filter_size; fi++) {
                            for (int fj = 0; fj < filter_size; fj++) {
                                sum += filters[n][d][fi][fj] * input[d][i + fi][j + fj];
                            }
                        }
                    }
                    output[n][i][j] = sum + biases[n];
                }
            }
        }
        return output;
    }

    // Backward propagation
    std::vector<std::vector<std::vector<float>>> backward(const std::vector<std::vector<std::vector<float>>>& d_out, float learning_rate) {
        int depth = input_data.size();
        int height = input_data[0].size();
        int width = input_data[0][0].size();

        int out_height = d_out[0].size();
        int out_width = d_out[0][0].size();

        // Initialize gradients
        std::vector<std::vector<std::vector<float>>> d_input(depth, std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));
        std::vector<std::vector<std::vector<std::vector<float>>>> d_filters(num_filters, std::vector<std::vector<std::vector<float>>>(depth, std::vector<std::vector<float>>(filter_size, std::vector<float>(filter_size, 0.0f))));
        std::vector<float> d_biases(num_filters, 0.0f);

        for (int n = 0; n < num_filters; n++) {
            for (int i = 0; i < out_height; i++) {
                for (int j = 0; j < out_width; j++) {
                    float delta = d_out[n][i][j];
                    // Update biases
                    d_biases[n] += delta;

                    for (int d = 0; d < depth; d++) {
                        for (int fi = 0; fi < filter_size; fi++) {
                            for (int fj = 0; fj < filter_size; fj++) {
                                int ni = i + fi;
                                int nj = j + fj;
                                // Update filter gradients
                                d_filters[n][d][fi][fj] += delta * input_data[d][ni][nj];
                                // Update input gradients
                                d_input[d][ni][nj] += delta * filters[n][d][fi][fj];
                            }
                        }
                    }
                }
            }
        }

        // Update weights and biases
        for (int n = 0; n < num_filters; n++) {
            biases[n] -= learning_rate * d_biases[n];
            for (int d = 0; d < depth; d++) {
                for (int fi = 0; fi < filter_size; fi++) {
                    for (int fj = 0; fj < filter_size; fj++) {
                        filters[n][d][fi][fj] -= learning_rate * d_filters[n][d][fi][fj];
                    }
                }
            }
        }

        return d_input;
    }
};

// ReLULayer class definition
class ReLULayer {
public:
    std::vector<std::vector<std::vector<float>>> input_data;

    // Forward propagation
    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input) {
        input_data = input;
        int depth = input.size();
        int height = input[0].size();
        int width = input[0][0].size();

        std::vector<std::vector<std::vector<float>>> output(depth, std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    output[d][i][j] = std::max(0.0f, input[d][i][j]);
                }
            }
        }
        return output;
    }

    // Backward propagation
    std::vector<std::vector<std::vector<float>>> backward(const std::vector<std::vector<std::vector<float>>>& d_out) {
        int depth = input_data.size();
        int height = input_data[0].size();
        int width = input_data[0][0].size();

        std::vector<std::vector<std::vector<float>>> d_input(depth, std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    if (input_data[d][i][j] > 0.0f) {
                        d_input[d][i][j] = d_out[d][i][j];
                    }
                }
            }
        }
        return d_input;
    }
};

// MaxPoolingLayer class definition
class MaxPoolingLayer {
public:
    int pool_size;
    int stride;
    std::vector<std::vector<std::vector<float>>> input_data;
    std::vector<std::vector<std::vector<std::pair<int, int>>>> max_indices;

    MaxPoolingLayer(int pool_size, int stride) {
        this->pool_size = pool_size;
        this->stride = stride;
    }

    // Forward propagation
    std::vector<std::vector<std::vector<float>>> forward(const std::vector<std::vector<std::vector<float>>>& input) {
        input_data = input;
        int depth = input.size();
        int height = input[0].size();
        int width = input[0][0].size();

        int out_height = (height - pool_size) / stride + 1;
        int out_width = (width - pool_size) / stride + 1;

        std::vector<std::vector<std::vector<float>>> output(depth, std::vector<std::vector<float>>(out_height, std::vector<float>(out_width, 0.0f)));
        max_indices.resize(depth, std::vector<std::vector<std::pair<int, int>>>(out_height, std::vector<std::pair<int, int>>(out_width)));

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < out_height; i++) {
                for (int j = 0; j < out_width; j++) {
                    float max_val = -INFINITY;
                    int max_i = -1, max_j = -1;
                    for (int pi = 0; pi < pool_size; pi++) {
                        for (int pj = 0; pj < pool_size; pj++) {
                            int ni = i * stride + pi;
                            int nj = j * stride + pj;
                            float val = input[d][ni][nj];
                            if (val > max_val) {
                                max_val = val;
                                max_i = ni;
                                max_j = nj;
                            }
                        }
                    }
                    output[d][i][j] = max_val;
                    max_indices[d][i][j] = std::make_pair(max_i, max_j);
                }
            }
        }
        return output;
    }

    // Backward propagation
    std::vector<std::vector<std::vector<float>>> backward(const std::vector<std::vector<std::vector<float>>>& d_out) {
        int depth = input_data.size();
        int height = input_data[0].size();
        int width = input_data[0][0].size();
        int out_height = d_out[0].size();
        int out_width = d_out[0][0].size();

        std::vector<std::vector<std::vector<float>>> d_input(depth, std::vector<std::vector<float>>(height, std::vector<float>(width, 0.0f)));

        for (int d = 0; d < depth; d++) {
            for (int i = 0; i < out_height; i++) {
                for (int j = 0; j < out_width; j++) {
                    auto max_idx = max_indices[d][i][j];
                    int ni = max_idx.first;
                    int nj = max_idx.second;
                    d_input[d][ni][nj] += d_out[d][i][j];
                }
            }
        }
        return d_input;
    }
};

// FullyConnectedLayer class definition
class FullyConnectedLayer {
public:
    int input_size;
    int output_size;
    std::vector<std::vector<float>> weights; // [output_size][input_size]
    std::vector<float> biases;
    std::vector<float> input_data;

    FullyConnectedLayer(int input_size, int output_size) {
        this->input_size = input_size;
        this->output_size = output_size;

        weights.resize(output_size, std::vector<float>(input_size, 0.0f));
        biases.resize(output_size, 0.0f);

        // Initialize weights with small random values
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                weights[i][j] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
            }
        }
    }

    // Forward propagation
    std::vector<float> forward(const std::vector<float>& input) {
        input_data = input; // Store input for backward pass
        std::vector<float> output(output_size, 0.0f);
        for (int i = 0; i < output_size; i++) {
            float sum = biases[i];
            for (int j = 0; j < input_size; j++) {
                sum += weights[i][j] * input[j];
            }
            output[i] = sum;
        }
        return output;
    }

    // Backward propagation
    std::vector<float> backward(const std::vector<float>& d_out, float learning_rate) {
        std::vector<float> d_input(input_size, 0.0f);
        // Update weights and biases
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                // Gradient w.r.t weights
                float d_weight = d_out[i] * input_data[j];
                weights[i][j] -= learning_rate * d_weight;
                // Accumulate gradient w.r.t input
                d_input[j] += weights[i][j] * d_out[i];
            }
            // Gradient w.r.t biases
            biases[i] -= learning_rate * d_out[i];
        }
        return d_input;
    }
};

// Sigmoid function
float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Derivative of sigmoid function
float sigmoid_derivative(float x) {
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
    std::string train_data_dir = "C:/Users/sahit/Downloads/archive/train";
    auto train_dataset = load_data(train_data_dir, image_height, image_width);

    // Load test data
    std::string test_data_dir = "C:/Users/sahit/Downloads/archive/test";
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

            // Backward pass

            // Gradient w.r.t output_fc2[0]
            float d_output_fc2 = d_loss_output * sigmoid_derivative(output_fc2[0]);

            // Backpropagate through fc2
            std::vector<float> d_output_fc2_vec = { d_output_fc2 };
            auto d_relu_fc1 = fc2.backward(d_output_fc2_vec, learning_rate);

            // Backpropagate through ReLU
            for (int i = 0; i < relu_fc1.size(); i++) {
                if (relu_fc1[i] <= 0.0f) {
                    d_relu_fc1[i] = 0.0f;
                }
            }

            // Backpropagate through fc1
            auto d_flattened_input = fc1.backward(d_relu_fc1, learning_rate);

            // Reshape d_flattened_input to match the shape of output_pool1
            std::vector<std::vector<std::vector<float>>> d_output_pool1(output_pool1.size(), std::vector<std::vector<float>>(output_pool1[0].size(), std::vector<float>(output_pool1[0][0].size(), 0.0f)));
            int idx = 0;
            for (int d = 0; d < output_pool1.size(); d++) {
                for (int i = 0; i < output_pool1[0].size(); i++) {
                    for (int j = 0; j < output_pool1[0][0].size(); j++) {
                        d_output_pool1[d][i][j] = d_flattened_input[idx++];
                    }
                }
            }

            // Backpropagate through pooling layer
            auto d_output_relu1 = pool1.backward(d_output_pool1);

            // Backpropagate through ReLU layer
            auto d_output_conv1 = relu1.backward(d_output_relu1);

            // Backpropagate through convolutional layer
            auto d_input = conv1.backward(d_output_conv1, learning_rate);

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
