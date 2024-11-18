from typing import Final

import requests
import streamlit as st

API_URL: Final[str] = "http://api:8000"


st.header("AI Assitant")
st.subheader("Upload a PDF & ask questions related to it.")

if "history" not in st.session_state:
    st.session_state.history = []

filename = None
file = st.file_uploader("File:", type="pdf")
upload_button = st.button("Upload PDF")

# Upload PDF Document and run RAG extraction/storage
if upload_button:
    file_upload_response = requests.post(
        f"{API_URL}/upload-pdf",
        files={"file": (file.name, file.getvalue(), "application/pdf")},
        headers={
            "accept": "application/json",
            # "Content-Type": "multipart/form-data",
        },
    )
    filename = file_upload_response.json()["filename"]
    st.write(f"{filename} has been processed.")

# AI Chat
if st.session_state.history:
    for msg in st.session_state.history:
        st.markdown(msg)

# Prompt
prompt = st.chat_input("Ask a question related to document.")
if prompt:
    prompt_response = requests.post(
        f"{API_URL}/prompt",
        headers={
            "accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "input": str(prompt),
        },
    )

    if prompt_response.ok:
        results = prompt_response.json()
        answer = results["answer"]
        filename = results["context"][0]["metadata"]["source"]
        source_pages = sorted([page["metadata"]["page"] for page in results["context"]])

        with st.chat_message("user"):
            st.write(prompt)
            st.session_state.history.append(prompt)
        with st.chat_message("assistant"):
            st.write(answer)
            st.write(f"Reference pages from {filename}:\n\t{source_pages}")
            st.session_state.history.append(f"{answer} (Pages: {source_pages})")
