import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import streamlit as st

import numpy as np

def adicionar_pdf(caminho_pdf):
    if caminho_pdf in st.session_state.chroma_docs:
        st.info(f"O documento '{caminho_pdf}' já foi processado anteriormente.")
        return  

    loader = PyPDFLoader(caminho_pdf)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    embeddings = OllamaEmbeddings(model="llama3")

    client = st.session_state.chroma_client
    collection = st.session_state.chroma_descricoes

    for doc in docs:
        embedding_vector = embeddings.embed_query(doc.page_content) 

        if not isinstance(embedding_vector, list) or not all(isinstance(x, float) for x in embedding_vector):
            st.error(f"Embedding inválido para o documento '{doc.metadata['title']}'. Esperado uma lista de floats.")
            return

        embedding_vector = np.array(embedding_vector, dtype=np.float32)

        try:
            collection.add(
                documents=[doc.page_content],
                metadatas=[{"pdf_name": caminho_pdf}],
                ids=[str(doc.metadata)],
                embeddings=[embedding_vector.tolist()] 
            )
            st.success(f"Documento '{doc.metadata['title']}' adicionado com sucesso!")
            break
        except Exception as e:
            st.error(f"Erro ao adicionar o documento '{doc.metadata['title']}': {e}")

    st.session_state.chroma_docs[caminho_pdf] = docs  
