# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 20:38:27 2023

@author: Welcome
"""

import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
# Import torch.nn.functional for applying softmax function to the output
import torch.nn.functional as F

# Load the saved model
# model = torch.load('model_quantized_and_trained.pth')

model = torch.jit.load('resnet_18_model_best_test_loss.pt')


# Define the image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Define the class labels
label_dict = {0: 'next', 1: 'pause', 2: 'play',3:'previous',4:'volume_down',5:'volume_up' }

# Set up the webcam
cap = cv2.VideoCapture(0)

# Loop over frames from the webcam
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    roi = frame[100:500, 100:500]
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    if not ret:
        break
    
    # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

    # Convert the frame to an image and apply the image transform
    image = Image.fromarray(frame)
    image = transform(image)
    image = image.unsqueeze(0)

    # Make a prediction using the trained model
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)[0]
        _, predicted = torch.max(output, 1)
        predicted_label = predicted.item()
        predicted_name = label_dict[predicted_label]
        print(probabilities)

    # Display the predicted gesture name and probability on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)
    thickness = 2
    x = 10
    y = 50
    cv2.putText(frame, f"{predicted_name}: {probabilities[predicted_label]:.2f}", (x, y), font, font_scale, color, thickness)

    # Show the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()