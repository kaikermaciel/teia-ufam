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
    check_system_resources
)
from config import MODEL_CHOICES, DEFAULT_RAM_LIMIT, DEFAULT_CPU_CORES, DEFAULT_GPU_LAYERS, UPLOAD_DIR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Função de logging
def log_to_file(model_name, ram_used, gpu_used, cpu_used, response_time, tokens, tokens_per_sec):
    """
    Função para registrar as informações de cada geração de resposta em um arquivo .txt.
    """
    log_data = f"""
    MODEL  : {model_name}
    RAM    : {ram_used} GB
    GPU    : {gpu_used} camadas
    CPU    : {cpu_used} núcleos
    TEMPO  : {response_time:.2f} s
    TOKEN  : {tokens}
    TOKEN/SEC : {tokens_per_sec:.2f}
    -------------------------------
    """
    
    with open("response_log.txt", "a") as log_file:
        log_file.write(log_data)

# Inicializa o estado da sessão
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.retriever = None
    st.session_state.documents = []  # Inicializa os documentos como uma lista vazia
if "question" not in st.session_state:
    st.session_state.question = ""  # Armazena a pergunta

def choose_model():
    st.title("Escolha o Modelo")
    model_choice = st.selectbox("Modelo", MODEL_CHOICES)
    ram = st.slider("Limite de RAM (GB)", 1, psutil.virtual_memory().available // (1024 ** 3), DEFAULT_RAM_LIMIT)
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

def get_relevant_documents(query, documents, threshold=0.2):
    """
    Função para recuperar documentos mais relevantes com base na consulta.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    document_vectors = vectorizer.fit_transform(documents)  # Vetoriza os documentos
    query_vector = vectorizer.transform([query])  # Vetoriza a consulta

    # Calcula a similaridade de cosseno entre a consulta e os documentos
    similarities = cosine_similarity(query_vector, document_vectors).flatten()

    # Ordena os documentos por relevância
    sorted_indices = similarities.argsort()[::-1]

    # Recupera os documentos com base no threshold de similaridade
    relevant_docs = []
    for i in sorted_indices:
        if similarities[i] >= threshold:
            relevant_docs.append((documents[i], similarities[i]))  # Adiciona o documento e a similaridade
        else:
            break
    print(len(relevant_docs), "documentos relevantes encontrados.")
    return relevant_docs

def query_ollama_model(model_name, user_input, documents):
    """
    Função para usar o modelo do Ollama para gerar uma resposta considerando documentos carregados.
    """
    # Recuperar documentos relevantes com base na consulta
    relevant_docs = get_relevant_documents(user_input, documents, threshold=0.2)
    
    # Montando o contexto para o Ollama incluir os documentos carregados
    messages = [{"role": "user", "content": user_input}]
    
    # Adiciona os documentos carregados como contexto
    for idx, (doc, _) in enumerate(relevant_docs):  # Agora passando documentos filtrados
        messages.append({"role": "system", "content": f"Documento {idx + 1}: {doc}"})  # Passando documentos como "system"
    
    # Medir o tempo de resposta
    start_time = time.time()
    response = ollama.chat(model=model_name, messages=messages)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    # Gerar a resposta com uma menção à origem dos dados (documentos)
    content = response['message']['content']
    # Concatenar as fontes (os documentos) ao final da resposta
    sources = "\n\nReferências dos documentos:\n"
    for idx, (doc, _) in enumerate(relevant_docs):
        sources += f"\nDocumento {idx + 1}: {doc[:200]}...\n"  # Mostra os primeiros 200 caracteres do documento como referência
    
    # Adiciona a fonte no final da resposta
    final_response = content + sources
    return final_response, processing_time

def query_and_measure(model, tokenizer, retriever, user_input,
                      max_new_tokens, min_length, top_p, temperature):
    inputs = tokenizer(user_input, return_tensors="pt")
    in_tokens = inputs["input_ids"].shape[1]

    # Se for RAG, recupera documentos
    if retriever:
        doc_scores, docs = retriever(inputs["input_ids"], return_tensors="pt")
        start = time.time()
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
        end = time.time()
    else:
        start = time.time()
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

    # Verifica a utilização de GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"O modelo será carregado na: {device}")
    
    # Verificar o uso da RAM
    ram_used_before = psutil.virtual_memory().used / (1024 ** 3)  # GB
    st.write(f"RAM usada antes de carregar o modelo: {ram_used_before:.2f} GB")
    
    if torch.cuda.is_available():
        st.write(f"Informações sobre a GPU: {torch.cuda.get_device_name(device)}")
        st.write(f"Uso de GPU: {torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} GB alocados.")
    
    # Limitar o número de núcleos da CPU
    os.environ["OMP_NUM_THREADS"] = str(cpu_cores)

    # Monitorando o uso do modelo
    start_time = time.time()

    # Simulação de carregamento de modelo
    model = torch.nn.Linear(100, 10)
    #model.to(device)
    #st.write("Modelo carregado com sucesso.")
    #st.write(f"Uso de GPU: {torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} GB alocados.")

    # Uso da memória após o carregamento
    ram_used_after = psutil.virtual_memory().used / (1024 ** 3)  # GB
    st.write(f"RAM usada após carregar o modelo: {ram_used_after:.2f} GB")
    
    if torch.cuda.is_available():
        st.write(f"Uso de GPU (max): {torch.cuda.memory_reserved(device) / (1024 ** 3):.2f} GB reservados.")
    
    end_time = time.time()
    execution_time = end_time - start_time
    st.write(f"Tempo de execução: {execution_time:.2f} segundos")
    
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
        for uf in uploaded_files:
            path = os.path.join(UPLOAD_DIR, uf.name)
            with open(path, "wb") as f:
                f.write(uf.getbuffer())
            txt = extract_text_from_pdf(path)
            st.session_state.documents.append(txt)
        st.success(f"{len(st.session_state.documents)} PDF(s) carregado(s).")

    # Contagem de tokens enquanto digita
    def count_tokens(text):
        return len(text.split()) // 2  # Aproximação: 2 palavras = 1 token

     # Botão para carregar modelo
    if st.button("Carregar Modelo"):
        check_system_resources(ram, cpu, gpu)
        with st.spinner("Carregando modelo..."):
            if model_choice == "deepseek-r1:1.5b":  # Novo modelo Ollama
                model_name = "deepseek-r1:1.5b"
                m = model_name  # Armazena o nome do modelo no estado da sessão
                tk = None  # Não usa tokenizer, pois Ollama é independente
                rt = None  # Não usa retriever no caso do Ollama
            elif  model_choice == "gemma3:4b":
                model_name = "gemma3:4b"
                m = model_name
                tk = None
                rt = None
            elif model_choice == "qwen3:4b":
                model_name = "qwen3:4b"
                m = model_name
                tk = None
                rt = None
            elif model_choice == "qwen3:8b":
                model_name = "qwen3:8b"
                m = model_name
                tk = None
                rt = None
            elif model_choice == "deepseek-r1:14b":
                model_name = "deepseek-r1:14b"
                m = model_name
                tk = None
                rt = None
            elif model_choice == "google/gemma-2-9B-it":
                m, tk = load_gemma2_model(model_choice)
                rt = None
            elif model_choice == "deepseek-ai/DeepSeek-R1-0528":
                m, tk = load_deepseek_model(model_choice)
                rt = None
            else:
                m, tk, rt = load_rag_model(model_choice)
                rt.n_docs = n_docs
        st.session_state.model = m
        st.session_state.tokenizer = tk
        st.session_state.retriever = rt
        st.success(f"Modelo {model_choice} pronto para uso.")

    # Área de entrada de texto e contagem de tokens enquanto digita
    if st.session_state.model:
        pergunta = st.text_area("Digite sua pergunta:", value=st.session_state.get("question", ""))
        tokens = count_tokens(pergunta)  # Contagem de tokens
        st.write(f"Tokens atuais: {tokens} tokens")

        # Área de pergunta e botão de envio
        if st.session_state.model:
            if st.button("Enviar Pergunta"):
                with st.spinner("Gerando resposta..."):
                    if st.session_state.model == "deepseek-r1:1.5b" or st.session_state.model == "gemma3:4b" or st.session_state.model == "qwen3:4b" or st.session_state.model == "qwen3:8b" or st.session_state.model == "deepseek-r1:14b":
                        # Consultar modelo Ollama diretamente, passando documentos carregados
                        response, tempo = query_ollama_model(st.session_state.model, pergunta, st.session_state.documents)
                    else:
                        if st.session_state.retriever and st.session_state.documents:
                            st.session_state.retriever.index_passages(st.session_state.documents)
                        response, tempo, tps = query_and_measure(
                            st.session_state.model,
                            st.session_state.tokenizer,
                            st.session_state.retriever,
                            pergunta,
                            max_new_tokens=256,
                            min_length=50,
                            top_p=0.9,
                            temperature=0.7
                        )
                st.write("**Resposta:**", response)
                if st.session_state.model != "deepseek-r1:1.5b":  # Só exibe para modelos RAG
                    st.write(f"Tempo de inferência: {tempo:.2f}s")
                    st.write(f"Tokens por segundo: {tps:.2f}")
                else:
                    # Exibe o tempo para o modelo DeepSeek
                    st.write(f"Tempo de resposta do Ollama: {tempo:.2f}s")
                
                tps = tokens/tempo
                log_to_file(st.session_state.model, ram, gpu, cpu, tempo, tokens, tps)


if __name__ == "__main__":
    main()
