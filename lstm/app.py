import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model  = load_model("next_word_lstm.h5")

with open("tokenizer.pickle",'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, max_sequence_len):
    # Convert the input text to a sequence of tokens
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Trim token list to fit the max sequence length
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    # Pad the sequence to match model input requirements
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    # Predict the next word index
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    # Map the predicted index back to the word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None

# Streamlit application
st.title("Next Word Prediction With LSTM and Early Stopping")

# Input text
input_text = st.text_input("Enter the sequence of words:", "To be or not to")

if st.button("Predict Next Word"):
    # Retrieve the max sequence length
    max_sequence_len = model.input_shape[1] + 1  # Add 1 for the next word prediction
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    
    # Display the predicted word
    if next_word:
        st.write(f'Next word: **{next_word}**')
    else:
        st.write("Could not predict the next word. Please try with a different sequence.")