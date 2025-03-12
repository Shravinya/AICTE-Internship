# AICTE-Internship
# AI Chatbot Using NLP & Machine Learning

This chatbot uses **Natural Language Processing (NLP)** and **Machine Learning (ML)** to respond to user queries.
## WEEK 1
## Features
- Uses `NLTK` for text processing.
- Implements `TF-IDF` for feature extraction.
- Trains a `Logistic Regression` model to classify user inputs.
- Prints `TF-IDF` vectors for analysis.
- Provides responses based on predefined intents.

## Improvements
- Added TF-IDF feature extraction to convert text into numerical form.
- Displayed TF-IDF vectors for better understanding of feature weights.
- Trained Logistic Regression model using TF-IDF features.


## WEEK 2
## Improvements
- Expanded intent matching with better response classification.
- Improved training data to enhance chatbot accuracy.
- Randomized responses for more natural conversations.
- Better handling of unknown queries to avoid incorrect replies.
- Refined intent labels to ensure accurate classification.


## WEEK 3
# Files Added
- app.py - Streamlit frontend for chatbot
- chat_log.csv - Stores conversation history
- chatbot.py - Backend logic for intent classification
- chatbot_model.h5 - Trained LSTM model
- classes.pkl - Serialized class labels for intent recognition
- intents.json - Dataset containing chatbot intents, patterns, and responses
- tokenizer.pkl - Serialized tokenizer for text processing


## Project Overview

- This AI-powered chatbot is designed to understand and respond to user queries using Natural Language Processing (NLP) and deep learning techniques. The chatbot leverages an LSTM-based neural network for intent classification and provides human-like responses.

# Features

- AI-powered response generation using an LSTM model.
- Natural Language Understanding (NLU) with tokenization and vectorization.
- Voice input and output support using speech recognition and text-to-speech (TTS).
- User-friendly interactive UI built with Streamlit.
- Conversation history storage and downloadable logs.

# Technologies Used
- Programming Language: Python
- Deep Learning: TensorFlow, LSTM
- NLP: NLTK, TF-IDF, Tokenization
- Web Framework: Streamlit
- Speech Processing: SpeechRecognition, pyttsx3

## 📂 Project Structure

- ├── app.py            # Streamlit frontend for chatbot
- ├── chatbot.py        # Backend logic for intent classification
- ├── chatbot_model.h5  # Trained LSTM model
- ├── classes.pkl       # Serialized class labels for intent recognition
- ├── tokenizer.pkl     # Serialized tokenizer for text processing
- ├── intents.json      # Dataset containing chatbot intents, patterns, and responses
- ├── chat_log.csv      # Stores conversation history
- ├── README.md         # Project documentation

## Installation & Setup

- Clone the repository
- git clone https://github.com/your-username/chatbot-project.git
- cd chatbot-project
- Install dependencies
- pip install -r requirements.txt
- Run the chatbot application
- streamlit run app.py

## How It Works

- The user inputs a message (text or voice).
- The message is tokenized and vectorized using TF-IDF.
- The LSTM model predicts the intent of the message.
- The chatbot selects an appropriate response from the intents.json file.
-The response is displayed in the UI and optionally read aloud.
- Chat history is stored and can be downloaded for review.

## Future Improvements

- Integrate Transformer-based models like BERT or GPT for improved accuracy.
- Implement context-awareness to remember previous messages.
- Deploy the chatbot as a web service using Flask or FastAPI.
- Enhance UI with real-time typing effects and custom themes.






