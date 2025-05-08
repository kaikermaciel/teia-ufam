import streamlit as st
import chromadb
from langchain_community.chat_message_histories import ChatMessageHistory

def init_session():
    if "chroma_docs" not in st.session_state:
        st.session_state.chroma_docs = {}
        st.session_state.descricoes_pdf = []
        st.session_state.chroma_descricoes = None
        
        try:
            st.write("Inicializando o cliente Chroma...")
            client = chromadb.Client()  # Cliente Chroma local
            st.write("Cliente Chroma inicializado com sucesso!")
            
            try:
                collection = client.get_collection("documents")
                st.write("Coleção 'documents' encontrada!")
            except chromadb.errors.NotFoundError:
                collection = client.create_collection("documents")
                st.write("Coleção 'documents' criada com sucesso!")
            
            st.session_state.chroma_client = client
            st.session_state.chroma_descricoes = collection 
            
        except Exception as e:
            st.error(f"Ocorreu um erro ao inicializar o cliente Chroma: {e}")
            st.error(f"Detalhes do erro: {e.__class__} - {str(e)}")
        
        st.session_state.chat_history = ChatMessageHistory()
        st.write("Histórico de conversa inicializado.")

