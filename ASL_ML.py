import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

BASE_PATH = 'EchoScribe ASL Dataset'

labels = []
data = []

for folder in os.listdir(BASE_PATH):
    folder_path = os.path.join(BASE_PATH, folder)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                image_path = os.path.join(folder_path, file_name)
                image = load_img(image_path, target_size=(128, 128))
                image_arr = img_to_array(image) / 255.0  # normalize pixel values
                data.append(image_arr)
                labels.append(folder)

data = np.array(data)
labels = np.array(labels)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(39, activation='softmax')  # 39 classes
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# import matplotlib.pyplot as plt
#
# base_dir = 'EchoScribe ASL Dataset'
# classes = ['H', 'E', 'L', 'O', 'nothing']
#
# transform = transforms.Compose([
#     transforms.Resize((150, 150)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# class_to_idx = {'H': 0, 'E': 1, 'L': 2, 'O': 3, 'nothing': 4}
# full_dataset = datasets.ImageFolder(base_dir, transform=transform, target_transform=lambda x: class_to_idx[x])
#
# train_size = int(0.8 * len(full_dataset))
# val_size = len(full_dataset) - train_size
# train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=False)
#
# # Debug: Print unique labels in the training dataset
# unique_labels = set()
# for _, labels in train_loader:
#     unique_labels.update(labels.numpy())
# print("Unique labels in training data:", unique_labels)
#
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(SimpleCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(128 * 18 * 18, 512),
#
#             nn.ReLU(inplace=True),
#             nn.Linear(512, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x
#
#
# model = SimpleCNN(num_classes=len(classes))
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0.0
#
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#
#         # Debug: Print unexpected labels
#         if (labels > 4).any():
#             print("Unexpected label detected during training:", labels)
#
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item() * inputs.size(0)
#
#     train_loss = train_loss / len(train_loader.dataset)
#     print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}')
#
# model.eval()
#
# correct = 0
# total = 0
#
# with torch.no_grad():
#     for inputs, labels in validation_loader:
#         outputs = model(inputs)
#
#         # Debug: Print unexpected labels during validation
#         if (labels > 4).any():
#             print("Unexpected label detected during validation:", labels)
#
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Validation Accuracy: {:.2f}%'.format(100 * correct / total))

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
#
# base_dir = 'EchoScribe ASL Dataset'
# classes = ['H', 'E', 'L', 'O', 'nothing']
#
# datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
#
# train_generator = datagen.flow_from_directory(
#     base_dir,
#     target_size=(150, 150),  # Resize images to 150x150
#     classes=classes,
#     class_mode='categorical',
#     subset='training'
# )
#
# validation_generator = datagen.flow_from_directory(
#     base_dir,
#     target_size=(150, 150),
#     classes=classes,
#     class_mode='categorical',
#     subset='validation'
# )
#
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dense(len(classes), activation='softmax')
# ])
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# history = model.fit(
#     train_generator,
#     epochs=10,
#     validation_data=validation_generator
# )
#
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()
