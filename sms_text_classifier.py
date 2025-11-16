import os
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = "sms_spam_model.h5"
TOKENIZER_PATH = "tokenizer"

if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH) and os.path.exists(TOKENIZER_PATH + ".index"):
    model = keras.models.load_model(MODEL_PATH)
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(TOKENIZER_PATH)
    max_len = int(open("max_len.txt").read())
else:
    train_df = pd.read_csv("train-data.tsv", sep="\t", header=None, names=["label","text"])
    valid_df = pd.read_csv("valid-data.tsv", sep="\t", header=None, names=["label","text"])

    train_df["label"] = train_df["label"].str.strip().str.lower().map({"ham":0, "spam":1})
    valid_df["label"] = valid_df["label"].str.strip().str.lower().map({"ham":0, "spam":1})

    x_train = train_df["text"].astype(str).tolist()
    y_train = train_df["label"].values
    x_valid = valid_df["text"].astype(str).tolist()
    y_valid = valid_df["label"].values

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(x_train, target_vocab_size=2**13)

    x_train_encoded = [tokenizer.encode(t) for t in x_train]
    x_valid_encoded = [tokenizer.encode(t) for t in x_valid]

    max_len = max(max(len(i) for i in x_train_encoded), max(len(i) for i in x_valid_encoded))
    open("max_len.txt", "w").write(str(max_len))

    x_train_padded = tf.keras.preprocessing.sequence.pad_sequences(x_train_encoded, maxlen=max_len)
    x_valid_padded = tf.keras.preprocessing.sequence.pad_sequences(x_valid_encoded, maxlen=max_len)

    class_weights = {}
    classes = np.unique(y_train)
    counts = np.bincount(y_train)
    total = len(y_train)
    for c in classes:
        class_weights[c] = total / (len(classes) * counts[c])

    model = keras.Sequential([
        keras.layers.Embedding(tokenizer.vocab_size, 32, input_length=max_len),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(x_train_padded, y_train, validation_data=(x_valid_padded, y_valid), epochs=10, class_weight=class_weights)

    model.save(MODEL_PATH)
    tokenizer.save_to_file(TOKENIZER_PATH)

def predict_message(message):
    encoded = tokenizer.encode(message)
    padded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=max_len)
    pred = model.predict(padded)[0][0]
    return [float(pred), "spam" if pred > 0.5 else "ham"]

print("\nType 'exit' to stop.\n")

while True:
    msg = input("Enter SMS: ")
    if msg.lower() == "exit":
        break
    print(predict_message(msg))
