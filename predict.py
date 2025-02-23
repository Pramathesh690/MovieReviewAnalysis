import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from src.util import SentimentAnalyzer

## Loading the model
model = load_model('simple_rnn_model.h5')
model.summary()



import streamlit as st

## streamlit app
## Streamlit app
st.title('IMDB Movie Review sentiment analysis')
st.write('Enter a movie review to classify its sentiment as positive or negative')

#user input
user_input = st.text_area('Movie Review')

if st.button('classify'):
  analyzer = SentimentAnalyzer(model,user_input , reverse_word_index)
 

  ## Make Prediction
  sentiment, probability = analyzer.predict(preprocess_input)

  st.write(f'Sentiment: {sentiment}')
  st.write(f'Predicition Score {prediction[0][0]}')

else:
  st.write("Please enter a movie review")