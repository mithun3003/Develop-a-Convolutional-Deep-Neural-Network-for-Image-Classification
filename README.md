# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET

### Problem Statement:

To develop a Convolutional Neural Network (CNN) for automatic image classification using the Fashion-MNIST dataset. The model extracts important features from clothing images using convolutional and pooling layers and classifies them into 10 different categories.

### Dataset: 
Fashion-MNIST

## Neural Network Model

<img width="2172" height="724" alt="12c3119a-0db0-41ba-81c6-4d05a518dbca" src="https://github.com/user-attachments/assets/eb20dd68-613f-478f-ad15-8f3085d15cf2" />

## DESIGN STEPS

### STEP 1: 
Import the required Python libraries, including PyTorch, Torchvision, NumPy, Matplotlib, Seaborn and Scikit-learn.

### STEP 2: 
Load the Fashion-MNIST dataset containing 60,000 training images and 10,000 testing images. Convert images into tensors and normalize the pixel values.

### STEP 3: 
Create DataLoaders with a batch size of 32 to load and process training and testing images efficiently.

### STEP 4: 
Design the CNN architecture using two convolutional layers, ReLU activation, max-pooling layers, a flatten layer and fully connected layers to classify 10 clothing categories.

### STEP 5: 
Initialize the CNN model, Cross Entropy Loss function and Adam optimizer. Train the model for 3 epochs using forward propagation and backpropagation.

### STEP 6:
Evaluate the trained model using test accuracy, confusion matrix and classification report. Predict the clothing category of a new sample image and compare the actual and predicted results.

## PROGRAM

### Name: MITHUN S

### Register Number: 212224240088

```python

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Name: MITHUN S")
print("Register Number: 212224240088")
print("Device:", device)



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])



train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)



test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

print("\nTraining Image Shape:", train_dataset[0][0].shape)
print("Training Dataset Size:", len(train_dataset))

print("\nTesting Image Shape:", test_dataset[0][0].shape)
print("Testing Dataset Size:", len(test_dataset))




train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)



class CNNClassifier(nn.Module):

    def __init__(self, input_size=1):

        super(CNNClassifier, self).__init__()

       

        self.conv_layers = nn.Sequential(

          

            nn.Conv2d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

           

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

       
        self.fc_layers = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),

            nn.Linear(128, 10)

        )

    def forward(self, x):

        x = self.conv_layers(x)

        x = self.fc_layers(x)

        return x




model = CNNClassifier(input_size=1).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

print("\nCNN Model Initialized Successfully")




def train_model(model, train_loader, num_epochs=3):

    model.train()

    epoch_losses = []

    for epoch in range(num_epochs):

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

           

            outputs = model(images)

          

            loss = criterion(outputs, labels)

           

            optimizer.zero_grad()

            loss.backward()

        

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        epoch_losses.append(epoch_loss)

        print("\nName: MITHUN S")
        print("Register Number: 212224240088")

        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Loss: {epoch_loss:.4f}"
        )

   

    plt.figure(figsize=(7, 5))

    plt.plot(
        range(1, num_epochs + 1),
        epoch_losses,
        marker='o'
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")

    plt.title("CNN Training Loss per Epoch")

    plt.xticks(range(1, num_epochs + 1))

    plt.grid()

    plt.show()

    return epoch_losses




epoch_losses = train_model(
    model,
    train_loader,
    num_epochs=3
)




def test_model(model, test_loader):

    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

           

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())

   

    accuracy = correct / total

    print("\n================================")

    print("Name: MITHUN S")
    print("Register Number: 212224240088")

    print(f"\nTest Accuracy: {accuracy:.4f}")

    print(f"Test Accuracy Percentage: {accuracy * 100:.2f}%")

    print("================================")


    

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=test_dataset.classes,
        yticklabels=test_dataset.classes
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("CNN Confusion Matrix")

    plt.xticks(rotation=45)

    plt.yticks(rotation=0)

    plt.tight_layout()

    plt.show()



    print("\nName: MITHUN S")

    print("Register Number: 212224240088")

    print("\nClassification Report:\n")

    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=test_dataset.classes
        )
    )




test_model(model, test_loader)




def predict_image(model, image_index, dataset):

    model.eval()

    image, label = dataset[image_index]

    with torch.no_grad():

        image_input = image.unsqueeze(0).to(device)

        output = model(image_input)

        _, predicted = torch.max(output, 1)

    class_names = dataset.classes

    predicted_label = predicted.item()



    print("\nName: MITHUN S")

    print("Register Number: 212224240088")

    plt.figure(figsize=(5, 5))

  

    display_image = image.squeeze().numpy() * 0.5 + 0.5

    plt.imshow(display_image, cmap="gray")

    plt.title(
        f"Actual: {class_names[label]}\n"
        f"Predicted: {class_names[predicted_label]}"
    )

    plt.axis("off")

    plt.show()



    print("\nNew Sample Data Prediction")

    print("Image Index:", image_index)

    print("Actual:", class_names[label])

    print("Predicted:", class_names[predicted_label])

    if label == predicted_label:

        print("Result: Correct Prediction")

    else:

        print("Result: Incorrect Prediction")




predict_image(
    model,
    image_index=80,
    dataset=test_dataset
)

```

### OUTPUT

## Training Loss per Epoch

<img width="796" height="1007" alt="image" src="https://github.com/user-attachments/assets/aee49f24-f8ea-4578-83b1-ad5d7bec5cf5" />

## Confusion Matrix

<img width="1170" height="1145" alt="image" src="https://github.com/user-attachments/assets/fc53dcc6-4378-4d7f-8b80-ef29d1b1f594" />

## Classification Report

<img width="540" height="472" alt="image" src="https://github.com/user-attachments/assets/b3a86188-984f-483f-8ccd-de3e79c6794c" />

### New Sample Data Prediction

<img width="676" height="761" alt="image" src="https://github.com/user-attachments/assets/26308cb0-7853-44b4-8cb6-34905b06331d" />

## RESULT
Thus, the Convolutional Neural Network (CNN) was successfully developed and implemented using PyTorch with the Fashion-MNIST dataset. The model was trained for 3 epochs using the Adam optimizer and Cross Entropy Loss function. The performance was evaluated using training loss, test accuracy, confusion matrix and classification report. Finally, the trained model successfully predicted the clothing category of a new sample image by comparing the actual and predicted labels.
