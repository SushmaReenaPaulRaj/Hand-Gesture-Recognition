# Hand Gesture Recognition for MP3 Player Using CNN  

## Overview  
This project implements a static hand gesture recognition system using **Convolutional Neural Networks (CNNs)** to control an MP3 player.  
The system recognizes six predefined hand gestures and maps them to music player commands such as play, pause, volume control, and track navigation.  

---

## Table of Contents  
- [Introduction](#introduction)  
- [Dataset](#dataset)  
- [Preprocessing](#preprocessing)  
- [Methodology](#methodology)  
- [Model Training](#model-training)  
- [Results](#results)  
- [Future Work](#future-work)  
- [Acknowledgements](#acknowledgements)  

---

## Introduction  
Hand gestures are a natural and intuitive way for humans to communicate. Traditional input methods like keyboards or touchscreens can be limiting, while gestures provide a **hands-free, expressive, and safe alternative**.  

This project demonstrates how CNNs can be applied to **hand gesture recognition** for real-world applications such as **controlling music playback**. The system uses **OpenCV** for image collection and preprocessing, and **PyTorch with ResNet-18** for training and classification.  

Supported commands include:  
- Play  
- Pause  
- Volume Up  
- Volume Down  
- Next  
- Previous  

---

## Dataset  
- **Custom dataset** of six gesture classes.  
- Each gesture contains **100 training images** and **50 testing images**.  
- Images were captured using a **webcam** in grayscale mode with OpenCV.  

**Gestures:**  
1. Play  
2. Pause  
3. Volume Up  
4. Volume Down  
5. Next  
6. Previous  

---

## Preprocessing  
- Resize images to **224×224 pixels**.  
- Convert to **grayscale** (reduce computational complexity).  
- Normalize with mean and standard deviation of **0.5**.  
- Apply **random horizontal flipping** (training only).  

---

## Methodology  

### Model Architecture  
- **ResNet-18** (pre-trained on ImageNet) used as the base model.  
- Fully connected layer replaced with a **custom classifier** for 6 gesture classes.  
- **Transfer Learning** applied with fine-tuning for improved accuracy.  

### Training Setup  
- Framework: **PyTorch**  
- Loss Function: **Cross-Entropy Loss**  
- Optimizer: **Stochastic Gradient Descent (SGD)** (lr=0.001, momentum=0.9)  
- Evaluation Metric: **Accuracy**  

---

## Model Training  

### Steps  
1. **Define transforms** for training and testing datasets.  
2. **Load datasets** with PyTorch’s `DataLoader`.  
3. Train the model with **5 epochs**, saving the best-performing model.  
4. **Evaluate on validation set** and generate a confusion matrix.  
5. Deploy for **real-time prediction** using webcam input.  

### Fine-Tuning  
- Pre-trained ResNet-18 weights were used.  
- Top layers unfrozen and retrained for better adaptation.  
- Custom classifier included:  
  - 2 Linear Layers  
  - 1 Dropout Layer  

---

## Results  

| Epoch | Training Loss | Training Accuracy | Test Loss | Test Accuracy |
|-------|---------------|-------------------|-----------|---------------|
| 1     | 0.8389        | 69.17%           | 0.0737    | 100%          |
| 2     | 0.2235        | 93.83%           | 0.0728    | 98%           |
| 3     | 0.1082        | 97%              | 0.0270    | 99%           |
| 4     | 0.1159        | 96.5%            | 0.0298    | 99%           |
| 5     | 0.1199        | 96.5%            | 0.0032    | 100%          |

✅ Achieved **~99–100% accuracy** on the test set.  
✅ Successfully recognized six distinct gestures in real-time.  

---

## Future Work  
- **Improve Generalization:** Train on larger, more diverse datasets.  
- **Real-Time Optimization:** Enhance speed and robustness for low-latency use.  
- **Scalability:** Extend to dynamic gestures and continuous gesture sequences.  
- **Applications:** Expand beyond MP3 control to **AR/VR, robotics, and accessibility tools**.   

---

## Acknowledgements  
- **OpenCV** for data collection and preprocessing.  
- **PyTorch** for deep learning implementation.  
- Pre-trained **ResNet-18** model contributors.  
- Research community for inspiring advancements in hand gesture recognition.  
