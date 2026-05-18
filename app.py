import streamlit as st
import sys
import os
import uuid

# Añadir la carpeta src al path para que Python encuentre nuestro agente
sys.path.append(os.path.abspath('src'))
from agent import responder_consulta

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
# Historial visual de la conversación
if "messages" not in st.session_state:
    st.session_state.messages = []

# ID único para la memoria de LangGraph (garantiza aislamiento de memoria por sesión)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# 4. Dibujar los mensajes guardados en la pantalla
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Capturar la entrada del usuario en la barra de chat inferior
if prompt := st.chat_input("Ej: Solicito información sobre el tiempo de respuesta para una urgencia alta..."):
    
    # Mostrar el mensaje del usuario en pantalla
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Guardarlo en la memoria visual de Streamlit
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 6. Procesar y mostrar la respuesta del Agente Funcional
    with st.chat_message("assistant"):
        with st.spinner("Procesando consulta y analizando base de conocimientos..."):
            # Llamamos al agente inyectando el ID de memoria único de esta sesión
            respuesta_agente = responder_consulta(
                pregunta=prompt, 
                thread_id=st.session_state.thread_id
            )
            st.markdown(respuesta_agente)
            
    # Guardar la respuesta del bot en la memoria visual de Streamlit
    st.session_state.messages.append({"role": "assistant", "content": respuesta_agente})