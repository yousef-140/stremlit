import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("HF_TOKEN")

API_URL = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

def query(user_message):
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": user_message}]
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()
    
    if "choices" in result:
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error: {result}"

st.title("AI Chat App")

user_input = st.text_input("Enter your question:")

if st.button("Generate"):
    if not user_input:
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            output = query(user_input)
        st.write(output)
