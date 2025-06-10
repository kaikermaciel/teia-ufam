import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import PyPDF2
import io
import os
import time

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="RAG com Gemma-2",
    page_icon="🤖",
    layout="wide"
)

# --- Cache do Modelo e Tokenizer ---
# O cache é refeito se qualquer um dos parâmetros de controle de recursos for alterado.
@st.cache_resource
def load_model(device_choice, max_vram_gb, cpu_threads):
    """
    Carrega o modelo Gemma-2 e o tokenizer do Hugging Face com controle de recursos.
    """
    # --- Controle de Threads da CPU ---
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        st.info(f"PyTorch configurado para usar no máximo {torch.get_num_threads()} threads de CPU.")

    model_name = "google/gemma-2-9b-it"
    max_memory_map = None
    
    # --- Lógica de Seleção de Dispositivo e Recursos ---
    if device_choice == 'Auto (GPU se disponível)' and torch.cuda.is_available():
        device = "cuda"
        device_map = "auto"
        torch_dtype = torch.bfloat16
        st.info(f"Carregando o modelo ({model_name}) em: GPU. Isso pode levar alguns minutos...")
        
        if max_vram_gb > 0:
            max_memory_map = {0: f"{max_vram_gb}GiB"} # Assume GPU no índice 0
            st.info(f"Tentando limitar a VRAM da GPU a {max_vram_gb}GiB. Camadas que não couberem serão movidas para a RAM.")
        
    else:
        device = "cpu"
        device_map = None 
        torch_dtype = torch.float32
        st.info(f"Carregando o modelo ({model_name}) em: CPU. Isso pode ser lento e consumir muita RAM...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory_map # Aplica o limite de memória
        )
        
        if device_map is None:
            model.to(device)

        st.success(f"Modelo carregado com sucesso no dispositivo: {device.upper()}!")
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        st.error("Isso pode ser causado por falta de memória (RAM ou VRAM). Tente aumentar os limites ou usar um modelo menor.")
        st.stop()


def extract_text_from_file(uploaded_file):
    """
    Extrai o texto de um arquivo carregado (.txt, .md, .pdf).
    """
    if uploaded_file is None:
        return ""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_bytes = uploaded_file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
            return text
        else:
            return uploaded_file.getvalue().decode("utf-8")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return ""

def generate_response(tokenizer, model, device, context, query, max_new_tokens):
    """
    Gera uma resposta usando o modelo Gemma-2 com base no contexto e na pergunta.
    """
    if not query:
        st.warning("Por favor, insira uma pergunta.")
        return ""

    messages = [
        {"role": "user", "content": f'Use o seguinte contexto para responder à pergunta. Se a resposta não estiver no contexto, diga "Não encontrei a resposta no documento."\n\n--- CONTEXTO ---\n{context}\n\n--- PERGUNTA ---\n{query}'}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response_text

# --- Interface do Usuário (UI) ---
st.title("📄🤖 Chat com Documentos usando Gemma-2")
st.markdown("""
Esta aplicação utiliza `gemma-2-9b-it` para responder perguntas com base em um arquivo (RAG), com controles avançados de recursos.
**A alteração de qualquer configuração de recursos reiniciará o modelo.**
""")

# Colunas para a interface
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Configurações de Recursos e IA")
    
    device_option = st.selectbox(
        "Escolha o dispositivo de processamento",
        ('Auto (GPU se disponível)', 'Forçar CPU'),
        index=0,
        help="'Auto' usará a GPU se disponível. 'Forçar CPU' usará apenas o processador."
    )

    max_vram_option = 0
    if device_option == 'Auto (GPU se disponível)':
        max_vram_option = st.number_input(
            "Limite de VRAM da GPU (GB)", 0, 128, 0, 4,
            help="Defina o máximo de VRAM que o modelo pode usar. 0 significa sem limite. O que não couber será alocado na RAM."
        )

    available_cpus = os.cpu_count() or 4 
    cpu_threads_option = st.number_input(
        "Limite de Threads da CPU", 1, available_cpus, available_cpus, 1,
        help=f"Defina o número máximo de núcleos de CPU que o PyTorch pode usar. Sua máquina tem {available_cpus} núcleos."
    )

    tokenizer, model, device = load_model(device_option, max_vram_option, cpu_threads_option)
    
    st.divider()

    uploaded_file = st.file_uploader(
        "Faça o upload do seu documento", type=['txt', 'md', 'pdf'],
        help="O conteúdo deste arquivo será usado como contexto."
    )
    
    # --- NOVO: Seletor para os tokens ---
    token_options = [30, 50, 100, 500, 1000, 10000]
    max_tokens = st.selectbox(
        "Limite Máximo de Tokens para Resposta",
        options=token_options,
        index=2, # Define 100 como padrão
        help="Escolha o número máximo de tokens que o modelo pode gerar na resposta."
    )
    
    query = st.text_area(
        "Faça sua pergunta aqui:", height=150,
        placeholder="Ex: Qual é o principal objetivo do documento?"
    )
    
    submit_button = st.button("Gerar Resposta", type="primary", use_container_width=True)

context = ""
if uploaded_file:
    with st.spinner("Processando o arquivo..."):
        context = extract_text_from_file(uploaded_file)
        with col2:
             with st.expander("Conteúdo do Documento (Primeiros 1000 caracteres)"):
                st.write(context[:1000] + "...")

with col2:
    st.header("Resposta do Modelo")
    response_placeholder = st.empty()
    time_placeholder = st.empty() # Placeholder para o tempo de resposta

    if submit_button:
        if not context:
            st.warning("Por favor, faça o upload de um arquivo para fornecer contexto.")
        elif not query:
            st.warning("Por favor, digite sua pergunta.")
        else:
            with st.spinner(f"🤖 Gemma-2 está pensando no dispositivo {device.upper()}..."):
                start_time = time.time()
                response = generate_response(tokenizer, model, device, context, query, max_tokens)
                end_time = time.time()
                duration = end_time - start_time
                
                response_placeholder.markdown(response)
                # --- NOVO: Exibe o tempo de resposta ---
                time_placeholder.success(f"Resposta gerada em {duration:.2f} segundos.")
    else:
        response_placeholder.info("A resposta do modelo aparecerá aqui.")
