import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Cargar la API Key desde el archivo .env
load_dotenv()

# Rutas relativas asumiendo que ejecutamos desde la raíz del proyecto
DATA_PATH = "data/manual_softtech.pdf"
CHROMA_PATH = "vector_store"

def main():
    print("--- Iniciando Módulo de Ingesta ---")
    
    # Validar que el PDF exista
    if not os.path.exists(DATA_PATH):
        print(f" Error: No se encontró el archivo en {DATA_PATH}")
        print("Por favor, asegúrate de colocar un PDF llamado 'manual_softtech.pdf' en la carpeta 'data/'.")
        return

    # 2. Cargar el PDF
    print(" Cargando el documento PDF...")
    loader = PyPDFLoader(DATA_PATH)
    documentos = loader.load()
    print(f" Documento cargado: {len(documentos)} páginas detectadas.")

    # 3. Fragmentación (Chunking) - Como indicamos en el Punto 3.2 del informe
    print(" Fragmentando el texto (Recursive Chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,       # Tamaño máximo de cada fragmento
        chunk_overlap=50,     # Traslape para no perder el contexto entre cortes
        length_function=len
    )
    chunks = text_splitter.split_documents(documentos)
    print(f" Texto dividido en {len(chunks)} fragmentos.")

    # 4. Vectorización y guardado en ChromaDB
    print(" Generando Embeddings y guardando en VectorDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Creamos la base de datos vectorial y la persistimos en la carpeta local
    db = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=CHROMA_PATH
    )
    print(f" ¡Éxito! Base de datos vectorial persistida en la carpeta '{CHROMA_PATH}'.")

if __name__ == "__main__":
    main()