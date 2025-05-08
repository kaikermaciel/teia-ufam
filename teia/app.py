import os
import streamlit as st
from core.loader import adicionar_pdf
from core.responder import responder
from state.session import init_session

st.markdown("<h1 style='text-align: center; color: #4CAF50;'> Gerador de questionarios de PDFs </h1>", unsafe_allow_html=True)

init_session()

if "chroma_docs" not in st.session_state:
    st.session_state.chroma_docs = {}

uploaded_file = st.file_uploader("Suba um PDF", type=["pdf"])
if uploaded_file:
    path = os.path.join("tmp", uploaded_file.name)
    os.makedirs("tmp", exist_ok=True)
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
    
    if uploaded_file.name not in st.session_state.chroma_docs:
        adicionar_pdf(path)
        st.session_state.chroma_docs[uploaded_file.name] = True  
        st.success(f"Documento '{uploaded_file.name}' adicionado com sucesso!")

if uploaded_file and uploaded_file.name in st.session_state.chroma_docs:
    pergunta = st.text_input("Sobre o que você quer o questionário:")
    if pergunta:
        if not st.session_state.chroma_descricoes:
            st.warning("Envie pelo menos um PDF primeiro.")
        else:
            resposta, origem = responder(pergunta)
            st.markdown(f"<strong>Resposta com base no PDF: _{origem}_</strong>")
            st.markdown(f"<p style='font-size: 18px; color: #333;'>{resposta}</p>", unsafe_allow_html=True)
print(uploaded_file)


if st.checkbox("Mostrar histórico de conversa"):
    for msg in st.session_state.chat_history.messages:
        role = "Usuário" if msg.type == "human" else "Chat"
        st.markdown(f"<p><strong>{role}:</strong> {msg.content}</p>", unsafe_allow_html=True)
