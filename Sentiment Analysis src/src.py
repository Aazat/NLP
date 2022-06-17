import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import pathlib
import shutil
import re
import string

url = "https://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = keras.utils.get_file(origin=url, untar=True,  cache_dir=".", cache_subdir="")
dataset_dir= os.path.join(os.path.dirname(dataset), 'aclImdb')
#os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir)
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

batch_size = 1024
seed = 123
train_ds = keras.utils.text_dataset_from_directory(train_dir, validation_split=0.2, batch_size = batch_size, 
                                                   subset="training", seed=seed)

val_ds = keras.utils.text_dataset_from_directory(train_dir, batch_size = batch_size, validation_split=0.2,
                                                 subset="validation", seed=seed)

test_ds = keras.utils.text_dataset_from_directory('aclImdb/test', batch_size= batch_size)                                                 

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

train_X = train_ds.map(lambda x,y: x);



                                                   
max_features = 10000
sequence_length = 200
vectorization_layer = keras.layers.TextVectorization(standardize= custom_standardization,
                                                     max_tokens= max_features,
                                                     output_mode = 'int',
                                                     output_sequence_length= sequence_length
                                                     )


vectorization_layer.adapt(train_X)   
vocab_size = len(vectorization_layer.get_vocabulary())

callbacks = [keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    min_delta=5e-2,
    verbose=1,
    restore_best_weights=True,
)]

embedding_dim = 64

model = keras.models.Sequential([
                                 vectorization_layer,
                                 keras.layers.Embedding(vocab_size, embedding_dim, name="embedding", mask_zero= True),
                                 keras.layers.Bidirectional(keras.layers.LSTM(128)),
                                 keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(0.001)),
                                 keras.layers.Dropout(0.2),
                                 keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(0.001)),
                                 keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss = keras.losses.BinaryCrossentropy(from_logits=False),
              metrics = ['accuracy'])

model.fit(train_ds,
          validation_data= val_ds,
          epochs=50,
          callbacks= callbacks)
          
model.evaluate(test_ds)

import subprocess 
from google.colab import files

def download_model(model_path, output_name="zipped_model"):
  model.save(model_path)
  output_name += ".zip"
  subprocess.run(["zip" , "-r" , os.path.join("/content", output_name), model_path ])
  files.download(output_name)        