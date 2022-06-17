import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import pickle
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = os.path.join(BASE_DIR, "django_charnn" ,"stateful_model_v2")
TOKENIZER_PATH= os.path.join(BASE_DIR, "tokenizer.pkl")
# TEXT_PATH = os.path.join(BASE_DIR, "Shakespeare_text")

# with open(TEXT_PATH) as f:
#   shakespear_text = f.read()

# tokenizer = keras.preprocessing.text.Tokenizer(char_level= True)
# tokenizer.fit_on_texts([shakespear_text])

# max_id = len(tokenizer.word_index)

with open(TOKENIZER_PATH, 'rb') as input:
    tokenizer = pickle.load(input)

max_id = len(tokenizer.word_index)

model = keras.models.load_model(MODEL_PATH)


def preprocess(text):
    X = np.array(tokenizer.texts_to_sequences(text)) - 1
    return tf.one_hot(X, max_id)


# def predict_char(text):
#     X_new = preprocess([text])
#     y_pred_array = model.predict(X_new)
#     max_ind = np.argmax(y_pred_array, axis=2)
#     return tokenizer.sequences_to_texts(max_ind + 1)[0][-1]

def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_prob = model.predict(X_new)[0, -1:, : ]
    log_norm = tf.math.log(y_prob)/temperature
    char_id = tf.random.categorical(log_norm, num_samples= 1) +1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]

def complete_text(text, n_chars=50, temperature=1):
    for i in range(n_chars):
        text += next_char(text, temperature)
    return text

