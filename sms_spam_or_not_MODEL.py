import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization, Dense, Input
from tensorflow.keras.models import Sequential

train_df = pd.read_csv('train-data.tsv', sep='\t')
test_df = pd.read_csv('valid-data.tsv', sep='\t')
#importing tsv file as dataframe one for training and one
#for testing
text_col = None
label_col = None
for col in train_df.columns:
    if "message" in col.lower() or "text" in col.lower():
        text_col = col
    if "label" in col.lower() or "class" in colm.lower():
        label_col = col

if text_col is None or label_col is None:
    raise ValueError("text_col and label_col are required")
train_texts = train_df[text_col].astype(str).tolist()
test_texts = test_df[text_col].astype(str).tolist()

def label_to_int(label):
    label = str(label).lower()
    if label == "spam":
        return 1
    else:
        return 0

y_train = np.array(label_to_int(l) for l in train_df[label_col])
y_test = np.array(label_to_int(l) for l in test_df[label_col])

max_tokens = 20000
max_len = 100
text_vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode='int',
    output_sequence_length=max_len,
)
text_vectorizer.adapt(train_texts)
x_train = text_vectorizer(np.array(train_texts))
x_test = text_vectorizer(np.array(test_texts))
model = Sequential([
    Input(shape=(max_len,)),
    keras.layers.Embedding(input_dim=max_tokens, output_dim=32,
                           mask_zero=False),
    keras.layers.GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=32,
    epochs=10
)
def predict_message(message):
    seq = text_vectorizer([message])
    prob_spam = float(model.predict(seq)[0][0])
    label = "spam" if prob_spam >= 0.5 else "ham"
    return [prob_spam, label]