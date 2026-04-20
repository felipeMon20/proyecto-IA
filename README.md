# SoftTech Solutions - Asistente de Soporte Técnico (RAG)

Este repositorio contiene la implementación de un pipeline RAG (Retrieval-Augmented Generation) diseñado para automatizar la resolución de consultas técnicas de Nivel 1 para el software SaaS "SoftTech Solutions".

El proyecto fue desarrollado para la asignatura **Ingeniería de Soluciones con IA (ISY0101)**.

## Arquitectura y Tecnologías
Para asegurar un despliegue eficiente y de bajo costo, la solución utiliza un enfoque híbrido con modelos locales y Open Source:
* **Orquestador:** LangChain.
* **Embeddings (Local):** HuggingFace (`all-MiniLM-L6-v2`) - Procesamiento 100% local y gratuito.
* **Base de Datos Vectorial:** ChromaDB.
* **Modelo de Lenguaje (LLM):** Llama 3.1 8B (vía Groq API) para latencia ultrabaja.
* **Frontend:** Streamlit.

## Requisitos Previos
* Python 3.10 o superior.
* Una API Key gratuita de [Groq](https://console.groq.com/).

## Instrucciones de Instalación

**1. Clonar el repositorio**
git clone https://github.com/felipeMon20/proyecto-IA.git
cd "proyecto-IA"

**2. Crear y activar el entorno virtual**
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

**3. Instalar las dependencias**
pip install -r requirements.txt
pip install langchain-chroma langchain-huggingface langchain-groq sentence-transformers

**4. Configurar las variables de entorno**
Crea un archivo llamado `.env` en la raíz del proyecto y agrega tu llave de Groq:
GROQ_API_KEY=gsk_tu_clave_aqui

## Ejecución del Sistema

El sistema se divide en dos fases principales:

**Fase 1: Ingesta de Datos (Vectorización)**
Para leer los manuales PDF en la carpeta `data/` y generar la base de datos vectorial local, ejecuta:
python src/ingest.py
*(Nota: La primera vez que se ejecute, descargará el modelo de embeddings de HuggingFace).*

**Fase 2: Interfaz de Chatbot**
Una vez creada la carpeta `vector_store/`, levanta la interfaz gráfica ejecutando:
streamlit run app.py
Esto abrirá automáticamente el navegador en `http://localhost:8501` con el asistente listo para responder preguntas basadas estrictamente en la documentación.

## Autores
* Felipe Monsalve
* Gabriel Bermar