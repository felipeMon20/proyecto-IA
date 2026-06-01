from langchain.tools import tool
from src.retriever import setup_retriever 
import datetime
import csv
import os
import uuid

# Inicializamos el retriever aquí afuera para que cargue la BD una sola vez
retriever_obj = setup_retriever()

# HERRAMIENTA 1: CONSULTA
@tool
def buscar_en_manuales(consulta: str) -> str:
    """
    Útil para buscar información técnica, configuraciones o flujos de trabajo 
    específicos en los manuales oficiales de SoftTech Solutions.
    Usa esta herramienta siempre que el usuario pregunte cómo hacer algo en el sistema.
    """
    documentos = retriever_obj.invoke(consulta)
    
    if not documentos:
        return "No se encontró información relevante en los manuales."
    
    contexto = "\n\n".join([f"Fuente ({doc.metadata.get('source', 'Desconocida')}):\n{doc.page_content}" for doc in documentos])
    return contexto

# HERRAMIENTA 2: RAZONAMIENTO / LÓGICA
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

# HERRAMIENTA 3: ESCRITURA E INTEGRACIÓN (CRM Simulado)
@tool
def redactar_escalamiento(resumen_problema: str) -> str:
    """
    Útil para escalar un problema al Nivel 2 y registrar el ticket en la base de datos (CRM). 
    Usa esta herramienta cuando no puedas resolver el problema del usuario 
    con la información de los manuales.
    """
    fecha_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Generar un ID de ticket corto y único (Ej: 8A4B92C1)
    ticket_id = str(uuid.uuid4())[:8].upper() 
    
    # Ruta del archivo CSV que actúa como base de datos
    ruta_crm = os.path.join("data", "tickets_escalados.csv")
    
    # Verificar si el archivo ya existe para incluir los encabezados
    archivo_existe = os.path.isfile(ruta_crm)
    
    try:
        # Escribir físicamente el ticket en el archivo
        with open(ruta_crm, mode='a', newline='', encoding='utf-8') as archivo:
            writer = csv.writer(archivo)
            if not archivo_existe:
                writer.writerow(["Ticket_ID", "Fecha", "Prioridad", "Estado", "Descripcion"])
            
            writer.writerow([ticket_id, fecha_actual, "Alta", "Abierto", resumen_problema])
            
        return f"El incidente ha sido escalado exitosamente a Nivel 2. El número de ticket asignado es {ticket_id}. Un ingeniero de soporte lo contactará a la brevedad."
    
    except Exception as e:
        return f"Error interno al intentar conectar con el CRM de SoftTech: {str(e)}"

# Lista de herramientas para exportar al Agente
herramientas_agente = [buscar_en_manuales, calcular_tiempo_respuesta, redactar_escalamiento]