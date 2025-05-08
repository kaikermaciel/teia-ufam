import streamlit as st
from .chain import criar_chain
from .LLM import qa_prompt

def responder(pergunta):
    result = st.session_state.chroma_descricoes.query(
        query_texts=[pergunta], n_results=4
    )
    st.write("Resultado da consulta ao Chroma:", result)

    if 'metadatas' in result and len(result['metadatas']) > 0:
        nome_pdf = result['metadatas'][0][0]["pdf_name"]
        st.write(f"PDF encontrado: {nome_pdf}")
    else:
        st.error("Não foi possível encontrar o nome do PDF nos resultados da consulta.")
        return "Erro na consulta", None
    
    document = st.session_state.chroma_docs.get(nome_pdf, [None])[0]
    if not document:
        st.error(f"Documento '{nome_pdf}' não encontrado nos dados.")
        return "Documento não encontrado", nome_pdf
    
    retriever = st.session_state.chroma_descricoes.as_retriever()

    chain = criar_chain(retriever)

    resposta = chain.invoke({
        "input": pergunta,
        "chat_history": st.session_state.chat_history.messages,
        "prompt": qa_prompt  
    })

    st.session_state.chat_history.add_user_message(pergunta)
    st.session_state.chat_history.add_ai_message(resposta["answer"])

    return resposta["answer"], nome_pdf
