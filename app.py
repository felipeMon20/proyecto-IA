import streamlit as st
import sys
import os
import uuid
import re
import pandas as pd

# Añadir la carpeta src al path para que Python encuentre nuestro agente
sys.path.append(os.path.abspath('src'))
from agent import responder_consulta

# --- PROTOCOLO DE PRIVACIDAD (PII FILTER) ---
def aplicar_filtro_privacidad(texto: str) -> str:
    texto = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[CORREO_CENSURADO]', texto)
    texto = re.sub(r'\b\d{1,2}\.?\d{3}\.?\d{3}[-][0-9kK]\b', '[RUT_CENSURADO]', texto)
    texto = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[TARJETA_CENSURADA]', texto)
    return texto

# ==========================================
# 1. CONFIGURACION VISUAL (Inyección CSS del diseño HTML proporcionado)
# ==========================================
st.set_page_config(
    page_title="SoftTech IA", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Inyectamos las variables y estilos exactos de tu HTML en Streamlit
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* Variables de tu diseño */
    :root {
        --bg: #0D0F12;
        --surface: #131620;
        --surface2: #1C1F2A;
        --border: #2A2D3A;
        --accent: #7C6FFF;
        --text: #E8E8F0;
        --muted: #9CA3AF;
        --user-bubble: #1E2235;
    }

    /* Fondo principal y fuente */
    .stApp {
        background-color: var(--bg);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }
    
    /* Ocultar header por defecto de Streamlit */
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* ESTILOS DE LA BARRA LATERAL (SIDEBAR) */
    [data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
        color: var(--text);
    }
    
    /* Botón principal (Nueva conversación) */
    .stButton > button {
        background-color: var(--accent);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 10px;
        font-weight: 500;
        width: 100%;
        transition: opacity 0.15s;
    }
    .stButton > button:hover {
        opacity: 0.85;
        background-color: var(--accent);
        color: white;
    }

    /* ESTILOS DEL CHAT */
    /* Input de chat */
    [data-testid="stChatInput"] {
        background-color: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px;
        color: var(--text);
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text) !important;
    }
    
    /* Burbujas de Chat */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    /* Contenedor del avatar */
    .stChatMessageAvatar {
        background: linear-gradient(135deg, #7C6FFF, #A78BFA);
        color: white;
        border-radius: 8px;
    }
    
    /* Estilo específico para el mensaje del usuario (Gris oscuro) */
    [data-testid="chatAvatarIcon-user"] {
        background-color: var(--surface3);
    }
    div[data-testid="stChatMessage"]:nth-child(even) {
        background-color: var(--user-bubble) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
    }

    /* Estilo de los tabs (Chat / Dashboard) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--bg);
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--muted);
        background-color: var(--surface);
        border-radius: 8px 8px 0 0;
        border: 1px solid var(--border);
        border-bottom: none;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--text);
        background-color: var(--surface2);
        border-top: 2px solid var(--accent);
    }
    
    /* Tarjetas del Dashboard */
    div[data-testid="stMetric"] {
        background-color: var(--surface2);
        border: 1px solid var(--border);
        padding: 15px;
        border-radius: 12px;
        color: var(--text);
    }
    [data-testid="stMetricValue"] {
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. BARRA LATERAL (SIDEBAR)
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
            <div style="width:28px; height:28px; border-radius:8px; background:linear-gradient(135deg,#7C6FFF,#A78BFA); display:flex; align-items:center; justify-content:center; font-weight:bold; color:white;">S</div>
            <h3 style="margin:0; font-size:16px;">SoftTech IA</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Nueva conversacion"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
        
    st.markdown("<div style='font-size:10px; color:#9CA3AF; text-transform:uppercase; margin-top:20px; margin-bottom:10px;'>Recientes</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#E8E8F0; font-size:13px; padding:8px; border-radius:8px; background:#1C1F2A; margin-bottom:4px;'>Error en modulo CRM</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#9CA3AF; font-size:13px; padding:8px;'>Falla en base de datos</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#9CA3AF; font-size:13px; padding:8px;'>Configuracion servidor</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top:auto; padding-top:20px; border-top:1px solid #2A2D3A;'><div style='font-size:12px; font-weight:bold;'>Operador N1</div><div style='font-size:10px; color:#9CA3AF;'>Soporte SoftTech</div></div>", unsafe_allow_html=True)

# ==========================================
# 3. AREA PRINCIPAL
# ==========================================
st.markdown("<h2 style='font-size:18px; font-weight:500; margin-bottom:20px; border-bottom:1px solid #2A2D3A; padding-bottom:10px;'>Asistente Tecnico | IA Nivel 1</h2>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

tab_chat, tab_dashboard = st.tabs(["Chat de Soporte", "Monitoreo del Sistema"])

# --- PESTAÑA: CHAT ---
with tab_chat:
    if len(st.session_state.messages) == 0:
        st.markdown("""
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:40px 0; color:#9CA3AF;">
                <div style="width:48px; height:48px; border-radius:14px; background:rgba(124,111,255,0.1); border:1px solid rgba(124,111,255,0.3); display:flex; align-items:center; justify-content:center; font-weight:bold; color:#A78BFA; margin-bottom:12px;">S</div>
                <h3 style="color:#E8E8F0; font-size:18px; margin:0 0 8px 0;">¿En que puedo ayudarte hoy?</h3>
                <p style="font-size:13px; margin:0;">Ingresa tu consulta tecnica o reporte de error.</p>
            </div>
        """, unsafe_allow_html=True)

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe tu consulta tecnica..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        prompt_seguro = aplicar_filtro_privacidad(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Procesando consulta..."):
                respuesta_agente = responder_consulta(
                    pregunta=prompt_seguro, 
                    thread_id=st.session_state.thread_id
                )
                st.markdown(respuesta_agente)
                
        st.session_state.messages.append({"role": "assistant", "content": respuesta_agente})
        st.rerun()

# --- PESTAÑA: DASHBOARD ---
with tab_dashboard:
    col1, col2, col3 = st.columns(3)
    col1.metric("Latencia Promedio", "1.6s", "-0.4s (Cache)")
    col2.metric("Consumo Tokens", "2,558", "-50% (Optimizacion)")
    col3.metric("Tasa Resolucion N1", "85%", "Estable")
    
    st.markdown("<br><h4 style='color:#E8E8F0; font-size:14px;'>Registro de Escalamientos (CRM)</h4>", unsafe_allow_html=True)
    
    ruta_crm = os.path.join("data", "tickets_escalados.csv")
    if os.path.exists(ruta_crm):
        try:
            df_tickets = pd.read_csv(ruta_crm)
            st.dataframe(df_tickets, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error al cargar CSV: {e}")
    else:
        st.info("No hay tickets escalados en el sistema.")