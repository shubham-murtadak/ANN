#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the MNIST dataset
data = tf.keras.datasets.mnist.load_data()

# Unpack the dataset into training and testing sets
(x_train, y_train), (x_test, y_test) = data

# Print shapes of training and testing data
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Normalize the pixel values of the images
x_train = x_train / 255
x_test = x_test / 255

# Reshape the images from 28x28 to 784-dimensional vectors
x_train = x_train.reshape(len(x_train), 28*28)
x_test = x_test.reshape(len(x_test), 28*28)

# Define the neural network model
model = Sequential([
    Dense(100, input_shape=(784,), activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model on the training data
model.fit(x_train, y_train, epochs=10)

# Evaluate the model on the testing data
model.evaluate(x_test, y_test)

# Make predictions on the testing data
y_pred = model.predict(x_test)
y_pred_label = [np.argmax(i) for i in y_pred]

# Print the first five predicted labels and true labels
print(y_pred_label[:5])
print(y_test[:5])

# Compute confusion matrix
cm = tf.math.confusion_matrix(y_test, y_pred_label)

# Plot the confusion matrix
plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predictions')
plt.ylabel('Truth')
plt.show()


#part B:evaluation of logistic regression
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build logistic regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes in the output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy for multi-class classification
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

# Predict probabilities on the test set
y_probs = model.predict(X_test)

# Convert probabilities to class predictions
y_pred = tf.argmax(y_probs, axis=1)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate precision score
precision = precision_score(y_test, y_pred, average='macro')  # Change 'macro' to 'micro' or 'weighted' if desired
print("Precision:", precision)

# Calculate recall score
recall = recall_score(y_test, y_pred, average='macro')  # Change 'macro' to 'micro' or 'weighted' if desired
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_test, y_pred, average='macro')  # Change 'macro' to 'micro' or 'weighted' if desired
print("F1-score:", f1)



# In[ ]:




