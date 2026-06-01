# SoftTech Solutions - Agente Funcional de Soporte Técnico Nivel 1

Este repositorio contiene la implementación de un Agente Inteligente basado en arquitectura LangGraph, diseñado para automatizar la resolución de consultas técnicas, tomar decisiones autónomas y mantener el contexto conversacional para el software SaaS "SoftTech Solutions".

El proyecto fue desarrollado para la asignatura **Ingeniería de Soluciones con IA (ISY0101)**.

## Arquitectura y Componentes del Agente
La solución ha evolucionado de un pipeline RAG tradicional a un Agente Funcional con capacidad de razonamiento ("Plan-and-Execute") preparado para entornos de producción. La arquitectura se compone de:

* **Orquestador (AgentExecutor):** Implementado con `langgraph.prebuilt.create_react_agent`, encargado de analizar la consulta y decidir la acción óptima.
* **Módulo de Memoria:** `MemorySaver` de LangGraph, implementado con gestión de `thread_id` para aislar y mantener el historial de conversaciones en flujos prolongados.
* **Modelo de Lenguaje (LLM):** Llama 3.3 70B Versatile (vía Groq API). Seleccionado específicamente por su alta capacidad de razonamiento lógico y precisión en la llamada a herramientas (Tool Calling).
* **Base de Datos Vectorial:** ChromaDB con Embeddings locales de HuggingFace (`all-MiniLM-L6-v2`). Limite de recuperación optimizado (`k=2`).

### Arsenal de Herramientas (Tools) e Integración
El agente posee autonomía para interactuar con el entorno mediante las siguientes herramientas:
1. `buscar_en_manuales`: Herramienta de recuperación semántica (RAG) que consulta la base de conocimientos vectorial.
2. `calcular_tiempo_respuesta`: Herramienta de razonamiento lógico que calcula Acuerdos de Nivel de Servicio (SLA).
3. `redactar_escalamiento`: Integración simulada con sistema CRM. Genera IDs únicos (UUID) y registra físicamente los incidentes críticos en la base de datos local (`data/tickets_escalados.csv`).

## Características de Producción (Unidad 3)
* **Observabilidad y Trazabilidad:** Integración nativa con LangSmith para el monitoreo de latencia (P50/P99), costo de tokens y análisis de trazas de ejecución en tiempo real.
* **Sostenibilidad y Escalabilidad (IA Verde):** Implementación de Caché Semántico en memoria (`InMemoryCache`) que reduce el tiempo de respuesta a 0ms y el costo de API a cero en consultas recurrentes.
* **Protocolos de Ciberseguridad y Privacidad:** * Middleware de sanitización mediante expresiones regulares (Regex) para la protección de PII (censura automática de RUTs, correos electrónicos y tarjetas de crédito).
  * Guardrails anti-Prompt Injection estructurados en el Prompt de Sistema para evitar manipulaciones y fuga de datos (Data Leakage).
* **Ciclo de Feedback:** Implementación de métricas de valoración del usuario en la interfaz gráfica para la mejora continua del sistema.

## Requisitos Previos
* Python 3.10 o superior.
* API Key de [Groq](https://console.groq.com/).
* API Key de [LangSmith](https://smith.langchain.com/) (Para telemetría).

## Instrucciones de Instalación y Ejecución

**1. Clonar el repositorio**
```bash
git clone [https://github.com/felipeMon20/proyecto-IA.git](https://github.com/felipeMon20/proyecto-IA.git)
cd "proyecto-IA"

2. Crear y activar el entorno virtual

python -m venv venv
venv\Scripts\activate

3. Instalar las dependencias

pip install -r requirements.txt

4. Configuración de Variables de Entorno (.env)
Crea un archivo .env en la raíz del proyecto con la siguiente estructura:

GROQ_API_KEY=gsk_tu_clave_aqui
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=[https://api.smith.langchain.com](https://api.smith.langchain.com)
LANGCHAIN_API_KEY=lsv2_tu_clave_aqui
LANGCHAIN_PROJECT=SoftTech_Soporte_Produccion

5. Fase de Ingesta (Preparación de Base de Datos)

python src/ingest.py

6. Despliegue de la Plataforma (Interfaz de Usuario)

streamlit run app.py

Autores
 Felipe Monsalve
 Gabriel Bermar