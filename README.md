# SoftTech Solutions - Agente Funcional de Soporte Técnico Nivel 1

Este repositorio contiene la implementación de un **Agente Inteligente** basado en arquitectura **LangGraph**, diseñado para automatizar la resolución de consultas técnicas, tomar decisiones autónomas y mantener el contexto conversacional para el software SaaS **"SoftTech Solutions"**.

El proyecto fue desarrollado para la asignatura **Ingeniería de Soluciones con IA (ISY0101)**.

---

## Arquitectura y Componentes del Agente

La solución ha evolucionado desde un pipeline RAG tradicional hacia un **Agente Funcional con capacidad de razonamiento (Plan-and-Execute)**, preparado para entornos de producción.

### Componentes principales

* **Orquestador (AgentExecutor):** Implementado con `langgraph.prebuilt.create_react_agent`, encargado de analizar la consulta y decidir la acción óptima.
* **Módulo de Memoria:** `MemorySaver` de LangGraph, implementado mediante gestión de `thread_id` para aislar y mantener el historial de conversaciones en flujos prolongados.
* **Modelo de Lenguaje (LLM):** Llama 3.3 70B Versatile (vía Groq API), seleccionado por su capacidad de razonamiento lógico y precisión en la llamada a herramientas (*Tool Calling*).
* **Base de Datos Vectorial:** ChromaDB con embeddings locales de Hugging Face (`all-MiniLM-L6-v2`) y recuperación optimizada (`k=2`).

---

## Arsenal de Herramientas (Tools)

El agente posee autonomía para interactuar con el entorno mediante las siguientes herramientas:

1. **`buscar_en_manuales`**

   * Herramienta de recuperación semántica (RAG) que consulta la base de conocimientos vectorial.

2. **`calcular_tiempo_respuesta`**

   * Herramienta de razonamiento lógico para el cálculo de Acuerdos de Nivel de Servicio (SLA).

3. **`redactar_escalamiento`**

   * Integración simulada con un sistema CRM.
   * Genera identificadores únicos (UUID).
   * Registra incidentes críticos en la base de datos local (`data/tickets_escalados.csv`).

---

## Características de Producción (Unidad 3)

### Observabilidad y Trazabilidad

Integración nativa con **LangSmith** para:

* Monitoreo de latencia (P50/P99).
* Análisis de consumo de tokens.
* Seguimiento de trazas de ejecución en tiempo real.

### Sostenibilidad y Escalabilidad (IA Verde)

Implementación de **caché semántico en memoria (`InMemoryCache`)**, reduciendo significativamente el tiempo de respuesta y el consumo de recursos en consultas recurrentes.

### Protocolos de Ciberseguridad y Privacidad

* Middleware de sanitización mediante expresiones regulares (Regex) para proteger información sensible (PII).
* Censura automática de:

  * RUTs
  * Correos electrónicos
  * Tarjetas de crédito
* Guardrails anti-*Prompt Injection* incorporados en el prompt de sistema para prevenir manipulaciones y fugas de información (*Data Leakage*).

### Ciclo de Feedback

Implementación de métricas de valoración del usuario en la interfaz gráfica para apoyar la mejora continua del sistema.

---

## Requisitos Previos

* Python 3.10 o superior.
* API Key de Groq.
* API Key de LangSmith (para telemetría).

---

## Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/felipeMon20/proyecto-IA.git
cd proyecto-IA
```

### 2. Crear y activar el entorno virtual

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Crear un archivo `.env` en la raíz del proyecto:

```env
GROQ_API_KEY=gsk_tu_clave_aqui

LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_tu_clave_aqui
LANGCHAIN_PROJECT=SoftTech_Soporte_Produccion
```

### 5. Ejecutar la fase de ingesta

Este paso genera la base de datos vectorial utilizada por el sistema RAG.

```bash
python src/ingest.py
```

### 6. Iniciar la aplicación

```bash
streamlit run app.py
```

---

## Estructura General del Proyecto

```text
proyecto-IA/
│
├── app.py
├── requirements.txt
├── .env
│
├── src/
│   ├── ingest.py
│   ├── agent.py
│   ├── tools.py
│   └── memory.py
│
├── data/
│   ├── manuales/
│   └── tickets_escalados.csv
│
└── README.md
```

---

## Autores

* Felipe Monsalve
* Gabriel Bermar
