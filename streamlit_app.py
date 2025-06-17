import streamlit as st
import requests

st.set_page_config(page_title="HR Resource Chatbot", layout="wide")
st.title("HR Resource Query Chatbot")
st.markdown("Ask a question like: *Find Python developers with 3+ years experience*")

# Input box
user_query = st.text_input("Enter your query:", placeholder="e.g., Who can work on a healthcare ML project?")
top_k = st.slider("Top matches", 1, 10, 3)

if st.button("Search") and user_query:
    with st.spinner("Thinking..."):
        try:
            response = requests.post("http://localhost:8000/chat", json={
                "query": user_query,
                "top_k": top_k
            })
            if response.status_code == 200:
                data = response.json()
                st.success("Here's what I found:")
                st.markdown(data["response"])
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
