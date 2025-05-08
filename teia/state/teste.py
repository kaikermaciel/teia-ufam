import chromadb

# Inicializa o cliente Chroma
client = chromadb.Client()

# Tenta criar uma coleção
try:
    collection = client.create_collection("test_documents")
    print("Coleção criada com sucesso!")
except Exception as e:
    print(f"Erro ao criar coleção: {e}")
