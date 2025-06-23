import streamlit as st
import time
import os
import fitz
import torch
import psutil
import ollama  # Importando a biblioteca Ollama
from loader import (
    load_deepseek_model,
    load_gemma2_model,
    load_rag_model,
    check_system_resources,
    load_ollama_model
)
from config import MODEL_CHOICES, DEFAULT_RAM_LIMIT, DEFAULT_CPU_CORES, DEFAULT_GPU_LAYERS, UPLOAD_DIR

# Inicializa o estado da sessão
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.retriever = None
if "documents" not in st.session_state:
    st.session_state.documents = []

def choose_model():
    st.title("Escolha o Modelo")
    model_choice = st.selectbox("Modelo", MODEL_CHOICES)
    ram = st.slider("Limite de RAM (GB)", 1, 64, DEFAULT_RAM_LIMIT)
    cpu = st.slider("Número de núcleos da CPU", 1, psutil.cpu_count(logical=False), DEFAULT_CPU_CORES)
    gpu = st.slider("Número de camadas da GPU", 1, 8, DEFAULT_GPU_LAYERS)
    n_docs = st.slider("Número de documentos RAG", 1, 10, 5)
    
    return model_choice, ram, cpu, gpu, n_docs

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def query_ollama_model(model_name, user_input):
    """
    Função para usar o modelo do Ollama para gerar uma resposta.
    """
    response = ollama.chat(model=model_name, messages=[user_input])
    return response["text"]

def query_and_measure(model, tokenizer, retriever, user_input,
                      max_new_tokens, min_length, top_p, temperature):
    inputs = tokenizer(user_input, return_tensors="pt")
    in_tokens = inputs["input_ids"].shape[1]
    
    start = time.time()
    
    if retriever:
        doc_scores, docs = retriever(inputs["input_ids"], return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            doc_scores=doc_scores,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            no_repeat_ngram_size=2,
            length_penalty=1.2,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            early_stopping=True
        )
    else:
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            no_repeat_ngram_size=2,
            length_penalty=1.2,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            early_stopping=True
        )
    
    end = time.time()
    out_tokens = outputs.shape[1]
    processing_time = end - start
    tps = (in_tokens + out_tokens) / processing_time if processing_time > 0 else 0
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response, processing_time, tps

def check_system_resources(ram_limit, cpu_cores, gpu_layers):
    # Verificar a quantidade de RAM disponível
    available_ram = psutil.virtual_memory().available / (1024 ** 3)  # GB
    if available_ram < ram_limit:
        st.warning(f"RAM insuficiente: {available_ram:.2f} GB disponíveis. O modelo pode não funcionar corretamente.")
    
    # Verificar a quantidade de núcleos da CPU
    available_cpu = psutil.cpu_count(logical=False)
    if available_cpu < cpu_cores:
        st.warning(f"Núcleos de CPU insuficientes: {available_cpu} disponíveis.")
    
    # GPU (simulação para limitar camadas; seria mais complexo com um gerenciador de GPU)
    if torch.cuda.is_available():
        available_gpu = torch.cuda.device_count()
        if available_gpu < gpu_layers:
            st.warning(f"Camadas da GPU insuficientes: {available_gpu} disponíveis.")
    else:
        st.warning("Nenhuma GPU disponível.")

def main():
    model_choice, ram, cpu, gpu, n_docs = choose_model()

    # Upload de PDFs para consulta
    uploaded_files = st.file_uploader(
        "Carregar PDFs para consulta",
        type="pdf",
        accept_multiple_files=True
    )
    if uploaded_files:
        st.session_state.documents = []
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)
        progress_bar = st.progress(0)
        for i, uf in enumerate(uploaded_files):
            path = os.path.join(UPLOAD_DIR, uf.name)
            with open(path, "wb") as f:
                f.write(uf.getbuffer())
            txt = extract_text_from_pdf(path)
            st.session_state.documents.append(txt)
            progress_bar.progress((i + 1) / len(uploaded_files))
        st.success(f"{len(st.session_state.documents)} PDF(s) carregado(s).")

    # Botão para carregar modelo
    if st.button("Carregar Modelo"):
        check_system_resources(ram, cpu, gpu)
        with st.spinner("Carregando modelo..."):
            try:
                # Carregar o modelo de acordo com a escolha
                if model_choice == "deepseek-r1:1.5b":
                    model_name = "deepseek-r1:1.5b"
                    model = load_ollama_model(model_name)
                    tokenizer = None
                    retriever = None
                elif model_choice == "google/gemma-2-9B-it":
                    model, tokenizer = load_gemma2_model(model_choice)
                    retriever = None
                elif model_choice == "deepseek-ai/DeepSeek-R1-0528":
                    model, tokenizer = load_deepseek_model(model_choice)
                    retriever = None
                else:
                    model, tokenizer, retriever = load_rag_model(model_choice)
                    retriever.n_docs = n_docs
                
                # Armazenando no estado da sessão
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.retriever = retriever
                st.success(f"Modelo {model_choice} carregado com sucesso.")
            except Exception as e:
                st.error(f"Erro ao carregar o modelo: {e}")

    # Área de pergunta e botão de envio
    if st.session_state.model:
        pergunta = st.text_area("Digite sua pergunta:")
        # sliders de geração
        max_new_tokens = st.slider("Máx tokens gerados", 50, 1024, 256)
        min_length = st.slider("Min comprimento resposta", 10, 256, 50)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9)
        temperature = st.slider("Temperatura", 0.1, 1.0, 0.7)
        if st.button("Enviar Pergunta"):
            if st.session_state.retriever and st.session_state.documents:
                st.session_state.retriever.index_passages(st.session_state.documents)
            with st.spinner("Gerando resposta..."):
                if st.session_state.model == "deepseek-r1:1.5b":
                    # Consultar modelo Ollama diretamente
                    response = query_ollama_model(st.session_state.model, pergunta)
                else:
                    response, tempo, tps = query_and_measure(
                        st.session_state.model,
                        st.session_state.tokenizer,
                        st.session_state.retriever,
                        pergunta,
                        max_new_tokens,
                        min_length,
                        top_p,
                        temperature
                    )
            st.write("**Resposta:**", response)
            if st.session_state.model != "deepseek-r1:1.5b":  # Só exibe para modelos RAG
                st.write(f"Tempo de inferência: {tempo:.2f}s")
                st.write(f"Tokens por segundo: {tps:.2f}")

if __name__ == "__main__":
    main()
