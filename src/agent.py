from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import retriever # Importamos la función de búsqueda que creaste antes

# 1. Cargar la API Key desde el archivo .env
load_dotenv()

def responder_consulta(pregunta):
    print("\n Iniciando Agente de IA...")
    
    # 2. Cargar el recuperador (Retriever)
    motor_busqueda = retriever.setup_retriever()
    
    # 3. Buscar información en tu base de datos (VectorDB)
    print(f" Buscando en los documentos la respuesta para: '{pregunta}'")
    documentos_recuperados = motor_busqueda.invoke(pregunta)
    
    # Extraer el texto de los documentos encontrados
    contexto = "\n\n".join([doc.page_content for doc in documentos_recuperados])
    
    # 4. Configurar el System Prompt (El mismo del informe de Word)
    template = """Eres un asistente de soporte técnico experto de Nivel 1 para el software SaaS de SoftTech Solutions. Tu objetivo es resolver las dudas operativas de los clientes corporativos de manera amable, profesional y concisa.

    REGLA CRÍTICA: Debes responder a la consulta del usuario basándote ÚNICA Y EXCLUSIVAMENTE en los fragmentos de contexto proporcionados a continuación. Tienes estrictamente prohibido utilizar conocimiento externo, inventar funciones o asumir soluciones que no estén explícitamente en el texto.

    INSTRUCCIONES:
    1. Analiza la pregunta del usuario y busca la respuesta en el [CONTEXTO RECUPERADO].
    2. Si la respuesta se encuentra en el contexto, redacta una solución paso a paso.
    3. Al final de tu respuesta, debes incluir obligatoriamente la referencia exacta de dónde obtuviste la información.
    4. Si el contexto proporcionado no contiene la información necesaria para responder a la pregunta, NO INVENTES NINGUNA RESPUESTA. Simplemente responde: 'Lamento no poder ayudarle con esto. No encuentro la información exacta en nuestra documentación actual. Transferiré su caso a un ejecutivo de Nivel 2.'

    [CONTEXTO RECUPERADO]:
    {context}

    [CONSULTA DEL USUARIO]:
    {question}
    """
    prompt = PromptTemplate.from_template(template)
    
    # 5. Configurar el LLM (Llama 3 a través de Groq)
    print(" Procesando respuesta con Llama 3 (Groq)...")
    # Temperature 0.0 evita que el modelo sea "creativo" (cero alucinaciones)
    llm = ChatGroq(temperature=0.0, model_name="llama-3.1-8b-instant")
    
    # 6. Unir todo y generar la respuesta
    chain = prompt | llm
    
    respuesta = chain.invoke({
        "context": contexto,
        "question": pregunta
    })
    
    return respuesta.content

def main():
    # Probamos con la misma pregunta del paso anterior
    pregunta_prueba = "¿Qué es el Modelo Lambda?"
    
    respuesta_final = responder_consulta(pregunta_prueba)
    
    print("\n" + "="*60)
    print(" RESPUESTA FINAL DEL AGENTE:")
    print("="*60)
    print(respuesta_final)
    print("="*60)

if __name__ == "__main__":
    main()