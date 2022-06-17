import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import string
import re
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.path.join(BASE_DIR, 'Final_Model')

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

# print(custom_standardization("some<br> input: string<br />"))
model= keras.models.load_model(MODEL_PATH, custom_objects={'custom_standardization' : custom_standardization})

# prediction = model.predict(["This is a very good review"])
# print(prediction)

def get_label(text):
  """Takes list/batches of text as inputs
  pass single text value as [single]"""
  predictions = model.predict(text)
  return ["Positive" if y > 0.5 else "Negative" for y in predictions], predictions[0].round(2)*100

if __name__ == "__main__":
  
  PositiveComments = ["Loved it, big fan", "Totally Recommend", "This movie was so good i cried", "Hands down one of the best ever"]
  NegativeComments = ["Worst Movie ever", "Boring movie of all time", "Utter waste of time", "Disgusting piece of shit"]

  print(get_label(PositiveComments))
  print(get_label(NegativeComments))