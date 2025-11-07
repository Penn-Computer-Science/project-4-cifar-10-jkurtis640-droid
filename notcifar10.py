 
import tensorflow as tf 
from keras.datasets import cifar10
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print(tf.__version__)##(sns.__version__)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
## sns.counterplot(x = y_train)
## plt.show()

## Check to make sure that there are NO values that are not a number 
print("Any NaN Training: ", np.isnan(x_train).any())
print("Any NaN Testing: ", np.isnan(x_test).any())
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') /255.0
input_shape = (32,32,3)

from keras.utils import to_categorical

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

class_labels = ["airplaine","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
batch_size = 120
num_classes = 10
epochs = 10

model = tf.keras.models.Sequential(
    [
       tf.keras.layers.Conv2D(64,(5,5), padding='same',activation='relu',input_shape=input_shape),
       tf.keras.layers.Conv2D(64, (3,3), padding='same',activation='relu',input_shape=input_shape),
       tf.keras.layers.MaxPool2D(),
       tf.keras.layers.Dropout(0.25),
       tf.keras.layers.Conv2D(64, (3,3), padding='same',activation='relu',input_shape=input_shape),
       tf.keras.layers.Conv2D(64, (3,3), padding='same',activation='relu',input_shape=input_shape),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(num_classes, activation='softmax')
    ]
)

model.compile(optimizer='adam' ,loss='categorical_crossentropy',metrics=['acc'])

history = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs, validation_split=0.1)

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'],color='b',label="Training loss ")
ax[0].plot(history.history['val_loss'],color='r',label="Validation loss ")
legend = ax[0].legend(loc='best',shadow=True)
ax[0].set_title("loss")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

ax[1].plot(history.history['acc'],color='b',label="Training Accuracy ")
ax[1].plot(history.history['val_acc'],color='r',label="Validation Accuracy ")
legend = ax[1].legend(loc='best',shadow=True)
ax[1].set_title("Accuracy")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Accuracy")

plt.tight_layout()
plt.show()

