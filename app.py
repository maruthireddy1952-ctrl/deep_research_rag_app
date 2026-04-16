import streamlit as st
import requests

st.title("Deep Research Assistant")

question = st.text_input("Ask a research question")

if st.button("Ask"):

    response = requests.post(
        "http://localhost:8000/ask",
        json={"question": question}
    )

    result = response.json()

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")

    for source in result["sources"]:
        st.write("-", source)

    st.subheader("Retrieval Debug Info")

    st.write("Retrieval Attempts:", result["retrieval_attempts"])

    if result["query_rewritten"]:
        st.write("Query Rewritten To:", result["query_rewritten"])