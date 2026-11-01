import streamlit as st
import requests

# Configure the Streamlit page
st.set_page_config(page_title="Local AI CRM", page_icon="🗃️", layout="centered")
st.title("🗃️ Local AI CRM Query Engine")

# Define your FastAPI backend URL
API_URL = "http://127.0.0.1:8000/v1/chat/completions"

# Initialize session state to store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display as code block if it's a JSON response, otherwise normal text
        if message["role"] == "assistant" and "{" in message["content"]:
            st.code(message["content"], language="json")
        else:
            st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask about your CRM data..."):
    
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare payload for the FastAPI endpoint
    payload = {
        "messages": st.session_state.messages,
        "model": "qwen-crm"
    }

    # Fetch and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Extracting CRM data..."):
            try:
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    raw_json_output = result["choices"][0]["message"]["content"]
                    
                    st.code(raw_json_output, language="json")
                    st.session_state.messages.append({"role": "assistant", "content": raw_json_output})
                else:
                    st.error(f"Backend Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect. Ensure your FastAPI backend is running via Uvicorn on port 8000.")