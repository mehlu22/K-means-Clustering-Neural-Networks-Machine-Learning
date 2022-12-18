# A simple neural network written in keras to classify the iris data set
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
iris_data = load_iris()

x = iris_data.data
y_ = iris_data.target.reshape(-1, 1) # Convert iris data to a single column

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)

# Split the data for testing and traning sets using the function below
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

# Building the model

model = Sequential()

model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))
model.add(Dense(10, activation='relu', name='fc2'))
model.add(Dense(3, activation='softmax', name='output'))

# Adam function that takes in learning rate to optimize differently each time
optimizer = Adam(lr=1.0)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# Training of the machine learning model
model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)

#Test on unknown data

results = model.evaluate(test_x, test_y)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))