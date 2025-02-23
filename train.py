import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN , Dense




## Load the IMDB Dataset
max_features = 10000 ## Voc_size
(X_train , y_train), (X_test,y_test) = imdb.load_data(num_words = max_features)


## Inspect a sample review and its label
sample_review = X_train[0]
sample_label = y_train[0]

print(f'Sample Review(as intergers): {sample_review}')
print(f'Sample Label (as interger): {sample_label}')

## Mapping of words index back to words(for understanding)
word_index = imdb.get_word_index()
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
reverse_word_index


### Trian the simple RNN
max_len = 500
model = Sequential()
model.add(Embedding(max_features,128,input_length = max_len)) ## Embedding layer
model.add(SimpleRNN(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

### Create an instance of Earlystopping callbacks
from tensorflow.keras.callbacks import EarlyStopping
earlystopping = EarlyStopping(monitor = 'val_loss' , patience = 5 , restore_best_weights = True)


history = model.fit(
    X_train , y_train , epochs = 10 , batch_size = 32,
    validation_split = 0.2 ,
    callbacks = [earlystopping]
)

## Save model file
model.save('simple_rnn_model.h5')



