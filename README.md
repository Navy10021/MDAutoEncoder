# 🌟 MDAuto-Encoder: Anomaly Detection for Malware 🌟

## Introduction

Anomaly Detection is a critical field in data analysis that aims to identify rare and abnormal points or patterns, known as anomalies or outliers, within normal data. The MDAuto-Encoder leverages deep learning techniques, specifically autoencoders, to effectively detect these anomalies using unsupervised learning methods.

For malware detection, we have designed two high-performance models: a CNN-based Auto-Encoder and a ResNet-based Auto-Encoder. These models have shown exceptional capabilities in accurately identifying and reconstructing patterns of normal data, making them highly effective for anomaly detection in malware datasets.

## 📚 Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Training Process](#training-process)
- [Detection Process](#detection-process)
- [Performance Metrics](#performance-metrics)
- [Visualization](#visualization)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 🧩 Architecture

The MDAuto-Encoder is designed with two primary architectures to enhance its anomaly detection capabilities:

### 1. CNN-Based Autoencoder

The Convolutional Neural Network (CNN)-based autoencoder leverages convolutional layers to capture spatial hierarchies in the input data. This architecture is particularly effective for image data, where local patterns and features are critical. The structure includes:

1. **Input Layer**: Receives the original image data.
2. **Convolutional Encoding Layers**: Multiple convolutional layers reduce the dimensionality of the input while extracting significant features.
3. **Bottleneck (Latent Space)**: A compressed representation containing the essential features of the data.
4. **Convolutional Decoding Layers**: Convolutional layers that reconstruct the data from the latent space representation.
5. **Output Layer**: Produces the reconstructed image, which should be as similar as possible to the input image.

### 2. ResNet-Based Autoencoder

The Residual Network (ResNet)-based autoencoder incorporates residual blocks to improve the training of deeper networks by mitigating the vanishing gradient problem. This architecture is advantageous for more complex data representations. The structure includes:

1. **Input Layer**: Receives the original data.
2. **Residual Encoding Blocks**: Residual blocks reduce the dimensionality and extract features while maintaining information flow through skip connections.
3. **Bottleneck (Latent Space)**: A compressed representation containing the essential features of the data.
4. **Residual Decoding Blocks**: Residual blocks that reconstruct the data from the latent space representation.
5. **Output Layer**: Produces the reconstructed data, ensuring it closely resembles the input data.

Both architectures are trained to minimize the reconstruction error, typically measured using Mean Squared Error (MSE).

## 🏋️‍♂️ Training Process

During the training phase, the MDAuto-Encoder is trained on non-anomalous (normal) data. The objective is to learn the normal patterns and features of the input data to accurately reconstruct it.

## 🔍 Detection Process

In the detection phase, the trained MDAuto-Encoder is used to reconstruct new data points. The reconstruction error for each data point is calculated, and data points with high reconstruction errors are considered anomalies.

## 📊 Performance Metrics
The performance of the anomaly detection model was evaluated by the AUC value of the ROC bulletproof. 
In Figure LEFT, the best performance is an AUC value of 1.0, indicating complete size change. Additionally, on the right side of the figure, the MDAuto-Encoder model indicates whether or not the handle is coded. These results mean that the proposed anomaly detection model maintains consistently high performance even under various lever values.
![image](https://github.com/Navy10021/MDAutoEncoder/assets/105137667/4bb1264e-5553-45d1-90eb-fca8f448d079)


## 📈 Visualization
A comparison between the original malware image and the image reconstructed by the MDAuto-Encoder model. The high similarity demonstrates the model's strong reconstruction capabilities and its potential for effective anomaly detection.
![image](https://github.com/Navy10021/MDAutoEncoder/assets/105137667/8ac3f901-0fd0-4210-807a-3341468bfc5f)


## 🚀 Usage

To use the MDAuto-Encoder for anomaly detection, follow these steps:

1. **Data Preparation**: Prepare your dataset, ensuring it includes labeled normal and anomalous data for evaluation purposes.
```python
$ python code/dataset.py
```
2. **Model Training**: Train the MDAuto-Encoder on the normal data.
```python
$ python code/train.py
```
3. **Anomaly Detection**: Use the trained model to reconstruct new data points and calculate the reconstruction error to detect anomalies.

## 🤝 Contributing

Contributions to the MDAuto-Encoder project are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
