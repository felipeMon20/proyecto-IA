from langchain.tools import tool
# Importamos tu función real
from src.retriever import setup_retriever
import datetime

# Inicializamos el retriever aquí afuera para que cargue la BD una sola vez
# y el agente responda rápido en cada iteración.
retriever_obj = setup_retriever()

# HERRAMIENTA 1: CONSULTA (Tu RAG actual transformado en herramienta)
@tool
def buscar_en_manuales(consulta: str) -> str:
    """
    Útil para buscar información técnica, configuraciones o flujos de trabajo 
    específicos en los manuales oficiales de SoftTech Solutions.
    Usa esta herramienta siempre que el usuario pregunte cómo hacer algo en el sistema.
    """
    # Usamos el objeto que creamos arriba con tu función
    documentos = retriever_obj.invoke(consulta)
    
    if not documentos:
        return "No se encontró información relevante en los manuales."
    
    # Unimos el contenido para que el agente lo lea, incluyendo las páginas
    contexto = "\n\n".join([f"Fuente ({doc.metadata.get('source', 'Desconocida')} - Pág {doc.metadata.get('page', 'N/A')}):\n{doc.page_content}" for doc in documentos])
    return contexto

# HERRAMIENTA 2: RAZONAMIENTO / LÓGICA (Ejemplo: Calculadora de SLA)
@tool
def calcular_tiempo_respuesta(prioridad: str) -> str:
    """
    Útil para calcular el tiempo máximo de respuesta (SLA) de un ticket de soporte
    basado en su nivel de prioridad (alta, media, baja).
    """
    prioridad = prioridad.lower().strip()
    if prioridad == "alta":
        return "SLA Crítico: El tiempo máximo de respuesta es de 2 horas."
    elif prioridad == "media":
        return "SLA Normal: El tiempo máximo de respuesta es de 12 horas."
    elif prioridad == "baja":
        return "SLA Estándar: El tiempo máximo de respuesta es de 24 horas."
    else:
        return "Prioridad no reconocida. Por favor clasifique como alta, media o baja."

# HERRAMIENTA 3: ESCRITURA (Escalado de Tickets)
@tool
def redactar_escalamiento(resumen_problema: str) -> str:
    """
    Útil para redactar un ticket formal de escalamiento a Nivel 2. 
    Usa esta herramienta cuando no puedas resolver el problema del usuario 
    con la información de los manuales.
    """
    fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    ticket = f"""
    --- TICKET DE ESCALAMIENTO (NIVEL 2) ---
    Fecha: {fecha_actual}
    Prioridad: Requiere revisión técnica avanzada
    Descripción del Problema: 
    {resumen_problema}
    Acción: Transferido automáticamente por Agente de Nivel 1.
    ----------------------------------------
    """
    return ticket

# Lista de herramientas para exportar al Agente
herramientas_agente = [buscar_en_manuales, calcular_tiempo_respuesta, redactar_escalamiento]