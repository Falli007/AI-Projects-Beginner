#Importing all the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.utilis import to_categorical
import matplotlib.pyplot as plt

#To load the MNIST dataset
(train_images, train_labels), (test_images, test_labels)= datasets.mnist.load_data()

#To preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

#Reshaping the images to (28, 28, 1) as they are grayscale images
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

#To convert labels to one-hot encoding
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

#To build the model
model = models.Sequential()

#First convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

#Second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

#Third convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#Flattening the output
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

#Output layer with 10 neruons for 10 classes
model.add(layers.Dense(10, activation='softmax'))

#To compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#To train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

#To evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_accuracy *100:.2f}%')

# To make predictions
predictions = model.predict(test_images)
print(f'Predicted class for first test image: {np.argmax(predictions[0])}')

# To visualize the first test image and its predicted class
plt.imshow(test_images[0].reshape(28, 28), cmap='gray')
plt.title(f'Predicted class: {np.argmax(predictions[0])}')
plt.axis('off')
plt.show()
