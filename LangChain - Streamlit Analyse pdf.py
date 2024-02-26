import streamlit as st
import os
import numpy as np
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

OPEN_AI_API_KEY = os.environ["OPENAI_API_KEY"]

st.title("Document Query Analyzer")

uploaded_file = st.file_uploader("Upload PDF file", type="pdf")

if uploaded_file is not None :
    # Save the uploaded file to the specified directory
    save_dir="C:\Users\pdevi\OneDrive\Desktop\"
    file_path = os.path.join(save_dir, "uploaded_pdf.pdf")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    embeddings_model = OpenAIEmbeddings(openai_api_key=OPEN_AI_API_KEY)

    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = Ollama(model="mistral")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    query = st.text_input("Enter your query:")
    if st.button("Analyze"):
        if query:
            result = chain.invoke(query)
            st.write("Analysis Result:")
            st.write(result)
        else:
            st.write("Please enter a query.")

