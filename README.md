![header](https://capsule-render.vercel.app/api?type=waving&color=0:C0C0C0,50:A9A9A9,100:808080&height=300&section=header&text=MDAutoEncoder&fontColor=696969&fontSize=95&fontAlignY=40&fontAlign=50&animation=fadeIn&fontStyle=stroke)

<div align="center">

# 🔗 MDAutoEncoder: Anomaly Detection AutoEncoder Model for Malware Detection

</div>


## 📑 Project Overview

Anomaly Detection is a critical field in data analysis that aims to identify rare and abnormal points or patterns, known as anomalies or outliers, within normal data. The **MDAutoEncoder** leverages deep learning techniques, specifically autoencoders, to effectively detect these anomalies using unsupervised learning methods.

For malware detection, we have designed two high-performance models: a CNN-based Auto-Encoder and a ResNet-based Auto-Encoder. These models have shown exceptional capabilities in accurately identifying and reconstructing patterns of normal data, making them highly effective for anomaly detection in malware datasets.

## 📚 Table of Contents

- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Training Process](#-training-process)
- [Detection Process](#-detection-process)
- [Performance Metrics](#-performance-metrics)
- [Visualization](#-visualization)
- [Usage](#-usage)
- [Paper & Related Project](#-paper--related-project)
- [Contributors](#-contributors)
- [License](#-license)

## 📊 Dataset

Our models are trained on diverse malware image datasets, designed to challenge and test the robustness of **MDAutoEncoder**:

- **MalwarePix-small**: 3,915 images, derived from 12,000 base samples across 9 classes, using undersampling to ensure equity.
- **MalwarePix-medium**: 13,254 images, enhanced with data augmentation for richer training data.
- **MalwarePix-large**: Our largest dataset with 26,478 images, providing the depth needed for comprehensive model training.

**You can download the dataset from the following Google Drive folder:**
📁 [Download MalwarePix](https://drive.google.com/drive/folders/1d6pnDUoJt7tDYyCFxDcDiTYa-2TdVn7N)

![image](https://github.com/Navy10021/MDAutoEncoder/assets/105137667/f78503a1-50a2-48ce-b2f9-3a13b3a7e187)


## 🧩 Architecture

The **MDAutoEncoder** is designed with two primary architectures to enhance its anomaly detection capabilities:

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

During the training phase, the **MDAutoEncoder** is trained on non-anomalous (normal) data. The objective is to learn the normal patterns and features of the input data to accurately reconstruct it.

![image](https://github.com/Navy10021/MDAutoEncoder/assets/105137667/862f457f-ff99-430e-9d9a-7d7c563e26d4)


## 🔍 Detection Process

In the detection phase, the trained **MDAutoEncoder** is used to reconstruct new data points. The reconstruction error for each data point is calculated, and data points with high reconstruction errors are considered anomalies.

![image](https://github.com/Navy10021/MDAutoEncoder/assets/105137667/97772f00-8f91-47f4-a3fe-fd3c270ca9d0)


For more information about model training and detection, see the Jupyter notebook in [notebooks/MDAutoEncoder.ipynb](notebooks/MDAutoEncoder.ipynb).


## 📊 Performance Metrics
The AUC value of the ROC bulletproof evaluated the performance of the anomaly detection model. 
In Figure LEFT, the best performance is an AUC value of 1.0, indicating complete size change. Additionally, in Figure RIGHT, the **MDAutoEncoder** model indicates whether or not the handle is coded. These results mean that the proposed anomaly detection model maintains consistently high performance even under various lever values.
![image](https://github.com/Navy10021/MDAutoEncoder/assets/105137667/4bb1264e-5553-45d1-90eb-fca8f448d079)


## 📈 Visualization
A comparison between the original malware image and the image reconstructed by the **MDAutoEncoder** model. The high similarity demonstrates the model's strong reconstruction capabilities and potential for effective anomaly detection.
![image](https://github.com/Navy10021/MDAutoEncoder/assets/105137667/8ac3f901-0fd0-4210-807a-3341468bfc5f)


## 🚀 Usage

To use the **MDAutoEncoder** for anomaly detection, follow these steps:

1. **Data Preparation**: Prepare your dataset, ensuring it includes labeled normal and abnormal data for evaluation purposes.
```python
$ python code/dataset.py
```
2. **Model Training**: Train the **MDAutoEncoder** on the normal data.
```python
$ python code/train.py
```
3. **Anomaly Detection**: Use the trained model to reconstruct new data points and calculate the reconstruction error to detect anomalies.
4. If you want to see all steps at once, run the code below.
```python
$ python code/mdautoencoder.py
```

## 📚 Paper & Related Project

- 📝 ***심층 신경망 기반 차세대 악성코드 탐지에 관한 연구: 악성코드 시각화 및 고성능 분류 / 이상 탐지 모델 개발***
- 📝 ***Deep Neural Networks for Next-Generation Malware Detection: Malware Visualization Techniques and High-Performance Classification / Anomaly Detection Models***
- 💻 ***Check out related project on GitHub***: [ MDDenseResNet: Enhanced Malware Detection Using Deep Neural Networks](https://github.com/Navy10021/MDDenseResNet)

## 👨‍💻 Contributors
- **Seoul National University Graduate School of Data Science (SNU GSDS)**
- Under the guidance of ***Navy Lee***

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
