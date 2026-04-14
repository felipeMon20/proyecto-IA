import streamlit as st
import sys
import os

# Añadir la carpeta src al path para que Python encuentre nuestro agente
sys.path.append(os.path.abspath('src'))
from agent import responder_consulta

# Configuración de la página web
st.set_page_config(page_title="SoftTech RAG Bot", page_icon="🤖")

st.title("🤖 Asistente de Soporte - SoftTech Solutions")
st.markdown("¡Hola! Soy el agente de soporte de Nivel 1. ¿En qué te puedo ayudar hoy basándome en nuestra documentación oficial?")

# Inicializar la memoria (historial de chat) en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

# Dibujar los mensajes guardados en la pantalla
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capturar la entrada del usuario en la barra de chat inferior
if prompt := st.chat_input("Escribe tu consulta operativa aquí..."):
    
    # 1. Mostrar el mensaje del usuario en pantalla
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Guardarlo en la memoria
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Mostrar la respuesta del asistente
    with st.chat_message("assistant"):
        # Un pequeño spinner de carga visual
        with st.spinner("Buscando en la documentación de SoftTech..."):
            # ¡Llamamos a tu código backend!
            respuesta = responder_consulta(prompt)
            st.markdown(respuesta)
    
    # Guardarlo en la memoria
    st.session_state.messages.append({"role": "assistant", "content": respuesta})