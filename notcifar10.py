import seaborn as sns 
import tensorflow as tf 
from keras.datasets import cifar10
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print(tf.__version__)(sns.__version__)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
sns.counterplot(x = y_train)
plt.show()

## Check to make sure that there are NO values that are not a number 
print("Any NaN Training: ", np.isnan(x_train).any())
print("Any NaN Testing: ", np.isnan(x_test).any())
