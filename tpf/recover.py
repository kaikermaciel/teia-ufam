from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

def get_relevant_documents(query, documents, top_n=5, threshold=0.2):
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
    
    return relevant_docs[:top_n]

def query_and_measure(model, tokenizer, retriever, user_input,
                    max_new_tokens, min_length, top_p, temperature):
    # Recuperar documentos relevantes com base na consulta
    relevant_docs = get_relevant_documents(user_input, retriever.documents, top_n=5, threshold=0.2)
    
    # Se não houver documentos relevantes, avise ao usuário
    if not relevant_docs:
        return "Nenhum documento relevante encontrado.", 0, 0

    # Se houver documentos relevantes, adicione ao retriever
    retrieved_documents = [doc[0] for doc in relevant_docs]  # apenas o texto do documento
    retriever.index_passages(retrieved_documents)  # Atualiza o índice com os documentos recuperados
    
    # Recuperar a resposta do modelo com os documentos relevantes
    inputs = tokenizer(user_input, return_tensors="pt")
    input_ids = inputs['input_ids']
    
    start_time = time.time()
    outputs = model.generate(input_ids=input_ids, doc_scores=retriever(input_ids), max_new_tokens=max_new_tokens, min_length=min_length)
    end_time = time.time()
    
    # Decodificar a resposta
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Calcular tempo de processamento e tokens por segundo
    processing_time = end_time - start_time
    num_tokens_input = input_ids.shape[1]
    num_tokens_output = outputs.shape[1]
    tps = (num_tokens_input + num_tokens_output) / processing_time if processing_time > 0 else 0
    
    return response, processing_time, tps
