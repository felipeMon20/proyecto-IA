# SoftTech Solutions - Agente Funcional de Soporte Técnico Nivel 1

Este repositorio contiene la implementación de un Agente Inteligente basado en arquitectura LangGraph, diseñado para automatizar la resolución de consultas técnicas, tomar decisiones autónomas y mantener el contexto conversacional para el software SaaS "SoftTech Solutions".

El proyecto fue desarrollado para la asignatura **Ingeniería de Soluciones con IA (ISY0101)**.

## Arquitectura y Componentes del Agente
La solución ha evolucionado de un pipeline RAG tradicional a un Agente Funcional con capacidad de razonamiento ("Plan-and-Execute"). La arquitectura se compone de:

* **Orquestador (AgentExecutor):** Implementado con `langgraph.prebuilt.create_react_agent`, encargado de analizar la consulta y decidir la acción óptima.
* **Módulo de Memoria:** `MemorySaver` de LangGraph, implementado con gestión de `thread_id` para aislar y mantener el historial de conversaciones en flujos prolongados.
* **Modelo de Lenguaje (LLM):** Llama 3.3 70B Versatile (vía Groq API). Seleccionado específicamente por su alta capacidad de razonamiento lógico y precisión en la llamada a herramientas (Tool Calling).
* **Base de Datos Vectorial:** ChromaDB con Embeddings locales de HuggingFace (`all-MiniLM-L6-v2`).

### Arsenal de Herramientas (Tools)
El agente posee autonomía para invocar las siguientes herramientas según las condiciones del entorno:
1. `buscar_en_manuales`: Herramienta de recuperación semántica (RAG) que consulta la base de conocimientos vectorial para responder dudas operativas.
2. `calcular_tiempo_respuesta`: Herramienta de razonamiento lógico que calcula Acuerdos de Nivel de Servicio (SLA) en base a la prioridad del incidente.
3. `redactar_escalamiento`: Herramienta de escritura que genera un reporte estructurado para derivar casos no resueltos al Nivel 2.

## Requisitos Previos
* Python 3.10 o superior.
* Una API Key de [Groq](https://console.groq.com/).

## Instrucciones de Instalación

**1. Clonar el repositorio**
```bash
git clone [https://github.com/felipeMon20/proyecto-IA.git](https://github.com/felipeMon20/proyecto-IA.git)
cd "proyecto-IA"
2. Crear y activar el entorno virtual

Bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate
3. Instalar las dependencias

Bash
pip install -r requirements.txt
pip install langchain-chroma langchain-huggingface langchain-groq sentence-transformers langgraph
4. Configurar las variables de entorno
Crea un archivo llamado .env en la raíz del proyecto y agrega tu llave de Groq:

Fragmento de código
GROQ_API_KEY=gsk_tu_clave_aqui
Ejecución del Sistema
El sistema opera en dos fases:

Fase 1: Ingesta de Datos (Preparación de la Base de Conocimientos)
Para procesar los manuales PDF en la carpeta data/ y generar la base de datos vectorial local, ejecuta:

Bash
python src/ingest.py
Fase 2: Despliegue de la Plataforma de Soporte
Para interactuar con el Agente Funcional a través de la interfaz gráfica, ejecuta:

Bash
streamlit run app.py
El sistema abrirá el navegador en http://localhost:8501. La plataforma inyectará un ID de sesión único automáticamente para gestionar la memoria a corto plazo del agente.

Autores
Felipe Monsalve
Gabriel Bermar


### ¿Qué mejoramos con este cambio?
* Se reemplazó el término "RAG" por "Agente Funcional" y "LangGraph", alineándose exactamente con el vocabulario técnico de la rúbrica (IE10).
* Se agregaron las 3 herramientas requeridas en la descripción (IE1).
* Se documentó el sistema de memoria `MemorySaver` y el `thread_id` (IE3 y IE4).
* Quedó un diseño sumamente sobrio y técnico.