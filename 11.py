#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

data=tf.keras.datasets.mnist.load_data()
(x_train,y_train),(x_test,y_test)=data
x_train.shape
plt.matshow(x_train[0])
y_train[0]

model=Sequential([
    Conv2D(30,(3,3),input_shape=(28,28,1),activation='relu'),
    MaxPool2D((2,2)),
    Flatten(),
    Dense(100,activation='relu'),
    Dense(10,activation='softmax')
    
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train,y_train,epochs=10)
model.evaluate(x_test,y_test)
y_pred=model.predict(x_test)
y_pred_label=[np.argmax(i) for i in y_pred]
y_pred_labels[:5]
y_test[:5]
cm=tf.math.confusion_matrix(y_pred_labels,y_test)
plt.figure(figsize=(7,7))
sns.heatmap(cm,annot=True,fmt='d')
plt.xlabel("Predictions")
plt.ylabel('Truth')

