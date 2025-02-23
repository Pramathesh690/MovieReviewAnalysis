import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

class SentimentAnalyzer:
    def __init__(self, model, text, reverse_word_index):
        self.model = model
        self.text = text
        self.reverse_word_index = reverse_word_index

    def decoded_review(self, encoded_review):
        """Decodes an encoded review into human-readable text."""
        return ' '.join([self.reverse_word_index.get(i - 3, '?') for i in encoded_review])

    def preprocess_text(self, text):
        """Preprocesses user input text for model prediction."""
        words = text.lower().split()
        encoded_review = [word_index.get(word, 2) + 3 for word in words]
        padded_reviews = sequence.pad_sequences([encoded_review], maxlen=500)
        return padded_reviews

    def predict_sentiment(self, review):
        """Predicts the sentiment of a given review."""
        preprocessed_input = self.preprocess_text(review)
        prediction = self.model.predict(preprocessed_input)
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        return sentiment, prediction

