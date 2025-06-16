import psutil
import streamlit as st
import time

import torch
from loader import load_rag_model, check_system_resources, load_documents_from_files
from config import MODEL_CHOICES, DEFAULT_RAM_LIMIT, DEFAULT_CPU_CORES, DEFAULT_GPU_LAYERS, UPLOAD_DIR
import os

# Função para mostrar a interface de escolha de modelo e configurações
def choose_model():
    st.title("Escolha o Modelo")

    # Seleção do modelo
    model_choice = st.selectbox("Escolha o Modelo", MODEL_CHOICES)

    # Configurações de hardware
    ram_limit = st.slider("Limite de RAM (GB)", 1, 64, DEFAULT_RAM_LIMIT)
    cpu_cores = st.slider("Número de núcleos da CPU", 1, psutil.cpu_count(logical=False), DEFAULT_CPU_CORES)
    gpu_layers = st.slider("Número de camadas da GPU", 1, 8, DEFAULT_GPU_LAYERS)

    return model_choice, ram_limit, cpu_cores, gpu_layers

# Função para fazer a consulta no modelo RAG
def query_rag_model(model, tokenizer, retriever, user_input):
    # Buscar documentos relevantes usando o retriever
    inputs = tokenizer(user_input, return_tensors="pt")
    input_ids = inputs['input_ids']
    doc_scores, docs = retriever(input_ids, return_tensors="pt")
    
    # Gerar resposta usando o modelo RAG
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, doc_scores=doc_scores)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    model_choice, ram_limit, cpu_cores, gpu_layers = choose_model()

    # Criação de diretório para upload
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    # Upload de arquivos para consulta
    uploaded_files = st.file_uploader("Carregar arquivos de consulta", type="txt", accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(UPLOAD_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Carregar documentos
        documents = load_documents_from_files(UPLOAD_DIR)
        st.success(f"{len(documents)} arquivos carregados com sucesso!")

    if st.button("Carregar Modelo"):
        try:
            check_system_resources(ram_limit, cpu_cores, gpu_layers)  # Verificar recursos do sistema
            start_time = time.time()
            model, tokenizer, retriever = load_rag_model(model_choice)  # Carregar o modelo RAG
            end_time = time.time()
            st.success(f"Modelo {model_choice} carregado com sucesso!")

            # Exibir o tempo de carregamento
            st.write(f"Tempo de carregamento: {end_time - start_time:.2f} segundos")

            # Entrada do usuário para consulta
            user_input = st.text_area("Digite sua consulta para o modelo:")

            if user_input:
                response_start_time = time.time()
                response = query_rag_model(model, tokenizer, retriever, user_input)
                response_end_time = time.time()
                st.write(f"Resposta do Modelo: {response}")
                st.write(f"Tempo de resposta: {response_end_time - response_start_time:.2f} segundos")

        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
    
if __name__ == "__main__":
    main()
