import streamlit as st
import sys
import os
import uuid
import re

# Añadir la carpeta src al path para que Python encuentre nuestro agente
sys.path.append(os.path.abspath('src'))
from agent import responder_consulta

# --- MEJORA IL3.3: PROTOCOLO DE PRIVACIDAD (PII FILTER) ---
def aplicar_filtro_privacidad(texto: str) -> str:
    """Detecta y anonimiza datos sensibles antes de que salgan al LLM externo."""
    # Censurar correos electrónicos
    texto = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[CORREO_CENSURADO]', texto)
    # Censurar RUT (formato con o sin puntos y guion)
    texto = re.sub(r'\b\d{1,2}\.?\d{3}\.?\d{3}[-][0-9kK]\b', '[RUT_CENSURADO]', texto)
    # Censurar números de tarjeta de crédito (16 dígitos)
    texto = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[TARJETA_CENSURADA]', texto)
    return texto

# 1. Configuración de la página web (Diseño Corporativo)
st.set_page_config(
    page_title="SoftTech Solutions | Soporte", 
    layout="centered"
)

# 2. Encabezado de la interfaz
st.title("Plataforma de Soporte Nivel 1")
st.subheader("SoftTech Solutions")
st.markdown("Bienvenido al sistema automatizado de asistencia corporativa. Ingrese su consulta técnica, dudas operativas o requerimientos de escalamiento a continuación.")
st.divider()

# 3. Inicializar variables de sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 4. Dibujar los mensajes y el Feedback (IL3.1 y IL3.2)
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Inyectar botones de feedback solo en las respuestas del sistema
        if message["role"] == "assistant":
            st.feedback("thumbs", key=f"feedback_{i}")

# 5. Capturar la entrada del usuario en la barra de chat inferior
if prompt := st.chat_input("Ej: Solicito información sobre el tiempo de respuesta para una urgencia alta..."):
    
    # Mostrar el mensaje original al usuario en la interfaz
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Aplicar el filtro de privacidad ANTES de procesar la lógica del agente
    prompt_seguro = aplicar_filtro_privacidad(prompt)

    # 6. Procesar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Procesando consulta bajo protocolos de seguridad..."):
            
            # El agente recibe el prompt ya filtrado
            respuesta_agente = responder_consulta(
                pregunta=prompt_seguro, 
                thread_id=st.session_state.thread_id
            )
            st.markdown(respuesta_agente)
            
    st.session_state.messages.append({"role": "assistant", "content": respuesta_agente})
    
    # Recargar la interfaz para dibujar el nuevo botón de feedback inmediatamente
    st.rerun()