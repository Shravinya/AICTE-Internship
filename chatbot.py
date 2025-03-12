import json
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load intents dataset
with open('intents.json') as file:
    data = json.load(file)

# Prepare training data
sentences, labels = [], []
classes = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern)
        labels.append(intent['tag'])
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
X = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(X, maxlen=20)

# Encode labels
label_map = {label: idx for idx, label in enumerate(classes)}
y = np.array([label_map[label] for label in labels])

# Build LSTM Model
model = Sequential([
    Embedding(5000, 128, input_length=20),
    SpatialDropout1D(0.2),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(len(classes), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X, y, epochs=100, batch_size=8, verbose=1)

# Save model and tokenizer
model.save("chatbot_model.h5")
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

print("Training complete! Model saved as chatbot_model.h5")













