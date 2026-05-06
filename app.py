import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

st.title("AI Chat App")

user_input = st.text_input("Enter your question:")

if st.button("Generate"):
    if not user_input:
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            output = query({"inputs": user_input})

        if isinstance(output, list):
            st.success(output[0].get("generated_text", output))
        elif "error" in output:
            st.error(f"Error: {output['error']}")
        else:
            st.write(output)