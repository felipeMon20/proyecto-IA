from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Rutas locales (las mismas que usamos para ingestar)
CHROMA_PATH = "vector_store"

def setup_retriever():
    """
    Función que carga la base de datos vectorial existente y configura el motor de búsqueda.
    """
    print(" Cargando modelo de embeddings local (HuggingFace)...")
    # Usamos exactamente el mismo modelo que usamos para vectorizar
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f" Cargando Base de Datos Vectorial desde: {CHROMA_PATH}...")
    # Cargamos ChromaDB apuntando a la carpeta donde guardamos los datos
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Configuramos el retriever para devolver los 3 chunks más relevantes (Top-K = 3)
    # tal como prometimos en el Punto 3.3 del informe
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    return retriever

def main():
    print("--- Probando Módulo de Recuperación (Retriever) ---")
    retriever = setup_retriever()
    
    # Aquí puedes escribir una pregunta relacionada con tu PDF de LinkedIn
    pregunta = "¿Qué es el Modelo Lambda?"
    
    print(f"\n Pregunta: '{pregunta}'\n")
    print(" Buscando en la base de datos vectorial...")
    
    # Ejecutamos la búsqueda semántica
    resultados = retriever.invoke(pregunta)
    
    print(f" Se encontraron {len(resultados)} fragmentos relevantes:\n")
    
    # Mostramos los resultados (Chunks) y sus metadatos (página de origen)
    for i, doc in enumerate(resultados):
        print(f"--- Resultado {i+1} ---")
        # El contenido real del PDF
        print(f"Texto: {doc.page_content.strip()}")
        # El metadato para la Trazabilidad (Punto 1.3 del informe)
        print(f"Fuente: {doc.metadata.get('source')} (Página {doc.metadata.get('page')})")
        print("-" * 30 + "\n")

if __name__ == "__main__":
    main()