 
import tensorflow as tf 
from keras.datasets import cifar10
import seaborn as sns 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

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

model.summary()

#predict the test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

#generate the confusion matrix
# Predict the values from the testing dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis=1) 
# Convert testing observations to one hot vectors
Y_true = np.argmax(y_test, axis=1)
# compute the confusion matrix
confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes) 

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Visualize the activations of each layer for a sample image
sample_image = x_test[0]  # Select a sample image from the test set
sample_image = np.expand_dims(sample_image, axis=0)  # Add batch dimension
# Create a model that outputs the activations of each layer
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs=model.inputs, outputs=layer_outputs)
# Get the activations for the sample image
activations = activation_model.predict(sample_image)
# Plot the activations for each layer
for layer_index, activation in enumerate(activations):
    num_filters = activation.shape[-1]
    size = activation.shape[1]

    
    # Create a grid to display the activations
    grid_size = int(np.ceil(np.sqrt(num_filters)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12,12))
    fig.suptitle(f'Layer {layer_index + 1} Activations', fontsize=16)

    for i in range(grid_size * grid_size):
        ax = axes[i // grid_size, i % grid_size]
        if i < num_filters:
          ax.imshow(activation[0, :, :, i], cmap='viridis')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
## Reflection
## 1. I noticed that mnist used on hot and required RMSprop while Cifar 10 used  the adam optimizer and used to_categrical.
## I also noticed that MNist was predictions based on numbers 0-9 while Cifar 10 was predictions based on physical things like airplane frog horse etc.
## 2. The changes that I did to increase performance was by not using seaborn and not overfitting my data by using less epochs.
## 3. If I had more time I would add pandas to this program to help with collecting data. 

## Finished 