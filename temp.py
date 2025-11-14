import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from AI.convolutionalNetworks import train_labels

data = pd.read_csv("insurance.csv")
label_encoder = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    #as our csv contains some columns which have string values we need
    #to convert this to numbers this for commands returns any column with
    #string value in it
    #LabelEncoder turn categorical into numeric labels
    #"male" -> 1
    #"female" -> 0
    data[col] = label_encoder.fit_transform(data[col])

train_dataset, test_dataset = train_test_split(data, test_size=0.2, random_state=42)
#split data into set of one 80% for training and 20% training
train_labels = train_dataset.pop("expenses")
test_labels = test_dataset.pop("expenses")
#We want to predict the expenses so we would not feed it to our model
#The expenses will be stored in labels
train_stats = train_dataset.describe().transpose()
#the describe() function tell every statistical detail about our data
#transpose flip rows with column
def normalize(x):
    return (x-train_stats["mean"])/train_stats["std"]
#z = (x-mean)/std
#by normalizing our model won't only focus on larger values but
#each value of feature is contributed equally
#after normalization each feature is in range of -3 to 3
norm_train_data = normalize(train_dataset)
norm_test_data = normalize(test_dataset)

def build_model():
    _model = keras.Sequential([
        #Sequential models layer are stacked one after another
        keras.Input(shape=(len(train_dataset.keys()),)),
        layers.Dense(64, activation='relu'),
        #Dense means every neuron is connected to every neuron in the next layer
        #input_shape tells how many numbers each training example has
        layers.Dense(64, activation='relu'),
        #Two Dense layer so our deep network could learn better
        layers.Dense(1)
    ])
    _model.compile(optimizer='adam', loss='mae', metrics=['mae', 'mse'])
    #MAE means absolute error (1/n)(actual-predicted)same unit as output
    #MSE Mean Squared Error (1/n)(actual-predicted)^2 Large mistakes punished more because of the square
    #metrics=['mae', 'mse'] show both during training
    return _model

model = build_model()
model.summary()
_epochs = 200
history = model.fit(
    norm_train_data, train_labels,
    epochs= _epochs, validation_split=0.2, verbose=0
    #validation_split means we take 20% of training data and use it for
    #performance testing not training
    #verbose = 0 show nothing on command line while training
    #verbose = 1 show progress bar
    #verbose = 2 show one line per epoch
)

loss, mae, mse = model.evaluate(norm_test_data, test_labels, verbose=2)
print(f"\nMean Absolute Error on test data: ${mae:.2f}")
predictions = model.predict(norm_test_data).flatten()