# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:07:36 2023

@author: Welcome
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import confusion_matrix

cudnn.benchmark = True
plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]),
}

data_dir = '/Users/Welcome/gesture_dataset'

def image_loader(path):
    return Image.open(path).convert("L")

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x], loader=image_loader)
                  for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5])
    std = np.array([0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])



def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf') # initialize best_loss to infinity

    

    for epoch in range(num_epochs):
      print('Epoch {}/{}'.format(epoch + 1, num_epochs))
      print('-' * 10)

      # Each epoch has a training and validation phase
      for phase in ['train', 'test']:
        if phase == 'train':
          model.train()  # Set model to training mode
        else:
          model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
          inputs = inputs.to(device)
          labels = labels.to(device)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward
          # track history if only in train
          with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            if phase == 'train':
              loss.backward()
              optimizer.step()

          # statistics
          running_loss += loss.item() * inputs.size(0)
          running_corrects += torch.sum(preds == labels.data)
        
        
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
          phase, epoch_loss, epoch_acc))

        
        # keep track of the best test loss
        if phase == 'test' and epoch_loss < best_loss:
           best_loss = epoch_loss
           best_acc = epoch_acc
           best_model_wts = copy.deepcopy(model.state_dict())

          
        if phase == 'test':
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            cm = confusion_matrix(all_labels, all_preds)

            # Define column names
            col_names = ['next', 'pause','play','previous','volume_down','volume_up']

            # Plot confusion matrix as an image
            plt.imshow(cm, cmap=plt.cm.Blues)

            # Add axis labels
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)

            # Add tick marks and labels for the x-axis
            tick_marks = np.arange(len(col_names))
            plt.xticks(tick_marks, col_names, fontsize=10, rotation = 90)

            # Add tick marks and labels for the y-axis
            plt.yticks(tick_marks, col_names, fontsize=10)

            # Add row and column labels
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i,j], ha='center', va='center', color='black', fontsize=12)

            # Display the confusion matrix
            plt.show()
            print(f'Confusion matrix:\n{cm}')
      
        if phase == 'test' and best_acc == epoch_acc:
           best_con_matrix = cm
                

      print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
      time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    print('Best test loss: {:4f}'.format(best_loss))
    print(f"Best confusion matrix:\n{best_con_matrix}")
   
    # Define column names
    col_names = ['next', 'pause','play','previous','volume_down','volume_up']

    # Plot confusion matrix as an image
    plt.imshow(best_con_matrix, cmap=plt.cm.Blues)

    # Add axis labels
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # Add tick marks and labels for the x-axis
    tick_marks = np.arange(len(col_names))
    plt.xticks(tick_marks, col_names, fontsize=10, rotation = 90)

    # Add tick marks and labels for the y-axis
    plt.yticks(tick_marks, col_names, fontsize=10)

    # Add row and column labels
    for i in range(best_con_matrix.shape[0]):
        for j in range(best_con_matrix.shape[1]):
            plt.text(j, i, best_con_matrix[i,j], ha='center', va='center', color='black', fontsize=12)

    # Display the confusion matrix
    plt.show()
    

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
        
        
model_ft = models.resnet18(models.ResNet18_Weights.DEFAULT)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 128),
                                nn.Dropout(),
                                nn.Linear(128, 6))
    
model_ft = model_ft.to(device)
    
criterion = nn.CrossEntropyLoss()
    
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    
    
    
model_ft = train_model(model_ft, criterion, optimizer_ft,
                           num_epochs=5)
    
visualize_model(model_ft)


resnet_18_model = torch.jit.script(model_ft)
resnet_18_model.save('resnet_18_model_best_test_loss.pt')