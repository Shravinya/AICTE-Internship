import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import json
import random
import os
import datetime
import pyttsx3
import speech_recognition as sr
import threading
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# 🌟 Streamlit Page Configuration
st.set_page_config(page_title="🤖 AI Chatbot", layout="wide", page_icon="💬")

# 🎯 Load AI Model and Assets
model = tf.keras.models.load_model("chatbot_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# 📂 Load Intents Data
with open('intents.json') as file:
    data = json.load(file)

# 🎙 Initialize Text-to-Speech
def speak(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# 🎧 Speech Recognition Function
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        with st.spinner("🎙 Listening..."):
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source)
                text = recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return "😕 Sorry, I couldn't understand."
            except sr.RequestError:
                return "🔴 Speech service unavailable."

# 🧠 Predict Intent
def predict_intent(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=20)
    prediction = model.predict(padded)
    intent = classes[np.argmax(prediction)]
    return intent

# 🤖 Generate Chatbot Response
def generate_response(user_input):
    intent = predict_intent(user_input)
    for i in data['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "❌ I'm sorry, I didn't understand that."

# 🔄 Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🚀 Main Function
def main():
    st.markdown("""
        <h1 style='text-align: center; color: #FF4B4B;'>💬 AI Chatbot 🤖</h1>
    """, unsafe_allow_html=True)

    menu = ["💬 Chat", "📜 Conversation History", "ℹ️ About"]
    choice = st.sidebar.radio("📌 Menu", menu)

    if choice == "💬 Chat":
        st.write("### 🚀 Let's Chat! Type or Speak 🎙")

        col1, col2 = st.columns([4, 1])
        user_input = col1.text_input("✍️ Type your message:")
        voice_input = col2.button("🎙 Speak")

        if voice_input:
            user_input = listen()
            st.success(f"🎤 You (Voice): **{user_input}**")

        if user_input:
            response = generate_response(user_input)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 📝 Store Conversation
            st.session_state.chat_history.append((user_input, response, timestamp))

            # 📢 Display Chat Messages
            for chat in st.session_state.chat_history:
                st.write(f"🧑‍💻 **You:** {chat[0]}")
                time.sleep(0.3)  # ⏳ Simulating Typing Effect
                st.markdown(f"🤖 **Chatbot:** `{chat[1]}`", unsafe_allow_html=True)
                st.text(f"🕒 {chat[2]}")
                st.markdown("---")

            speak(response)  # 🎙 Speak Response

            # 👋 Exit on Goodbye
            if response.lower() in ['goodbye', 'bye']:
                st.success("👋 Thank you for chatting! Have a great day! 😊")
                time.sleep(2)
                st.stop()

    elif choice == "📜 Conversation History":
        st.header("📜 Chat History")
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.write(f"🧑‍💻 **You:** {chat[0]}")
                st.markdown(f"🤖 **Chatbot:** `{chat[1]}`", unsafe_allow_html=True)
                st.text(f"🕒 {chat[2]}")
                st.markdown("---")

            # 💾 Download Chat History
            if st.button("💾 Download Chat History"):
                with open("chat_history.txt", "w") as file:
                    for chat in st.session_state.chat_history:
                        file.write(f"You: {chat[0]}\nBot: {chat[1]}\nTime: {chat[2]}\n---\n")
                st.success("📥 Chat history saved successfully!")

        else:
            st.warning("📭 No conversation history found.")

    elif choice == "ℹ️ About":
        st.header("ℹ️ About This Chatbot")
        st.write("""
        - 🤖 **AI-powered chatbot** built with deep learning (TensorFlow).  
        - 🧠 Uses NLP to understand and respond to queries.  
        - 🎙 Supports **voice input & text-to-speech responses**.  
        - 📜 **Stores conversation history** for review and download.  
        - 💡 **Real-time chat UI with typing effect & speech recognition.**  
        """)

if __name__ == '__main__':
    main()
