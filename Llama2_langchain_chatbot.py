#!/usr/bin/env python
# coding: utf-8

# In[3]:

!pip install langchain
!pip install replicate
!pip install langchain_community
# Importation des bibliothèques et modules nécessaires
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.llms import Ollama
import replicate
import os

# Définition d'un modèle de prompt pour la conversation
prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Vous êtes un influenceur Instagram et vous partagez beaucoup de choses sur votre vie avec les autres. Vous avez
    actuellement une conversation avec un être humain. Répondez aux questions
    avec un ton amical et sympathique, avec une touche d'humour.
    
    chat_history: {chat_history},
    Humain : {question}
    IA:"""
)

# Création d'instances de modèles et de mémoire
#Définition du modèle utilisé
llm = Ollama(model="llama2",temperature=0.2)

memory = ConversationBufferWindowMemory(memory_key="chat_history", k=4)
llm_chain = LLMChain(
    llm=llm,
    memory=memory,
    prompt=prompt
)
# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot" ,page_icon="🦙",
    layout="wide")

# Configuration du titre de l'application Streamlit
st.title("ChatGPT Clone")

with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Vérification de l'existence de messages dans la session et création s'ils n'existent pas
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Salut, je suis le clone de ChatGPT"}
    ]

# Affichage de tous les messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# # Collecte de l'entrée de l'utilisateur
# user_prompt = st.chat_input()

# Ajout de l'entrée utilisateur aux messages de la session
if user_prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

# Génération d'une réponse IA si le dernier message est de l'utilisateur
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Chargement..."):
            ai_response = llm_chain.predict(question=user_prompt)
            st.write(ai_response)
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)


