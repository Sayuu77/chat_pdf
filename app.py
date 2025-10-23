import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="RAG System - Asistente Inteligente",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema morado moderno
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a0b2e 0%, #2d1b69 50%, #4c1d95 100%);
        color: #f3f4f6;
    }
    .main-title {
        font-size: 2.8rem;
        text-align: center;
        background: linear-gradient(45deg, #a855f7, #c084fc, #e879f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-weight: 800;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .subtitle {
        text-align: center;
        color: #d8b4fe;
        margin-bottom: 2rem;
        font-size: 1.2rem;
    }
    .upload-card {
        background: rgba(255, 255, 255, 0.08);
        border: 2px dashed #a855f7;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    .upload-card:hover {
        background: rgba(168, 85, 247, 0.15);
        border-color: #c084fc;
    }
    .response-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 6px solid #a855f7;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
    .stButton button {
        background: linear-gradient(45deg, #a855f7, #c084fc);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.4);
    }
    .api-input {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid #a855f7;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem;
        border-left: 4px solid #c084fc;
    }
    .chat-bubble {
        background: linear-gradient(135deg, #a855f7, #c084fc);
        color: white;
        border-radius: 18px 18px 18px 4px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        max-width: 80%;
        box-shadow: 0 4px 12px rgba(168, 85, 247, 0.3);
    }
    .user-question {
        background: rgba(255, 255, 255, 0.15);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        margin-left: auto;
        max-width: 80%;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .sidebar-content {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<div class="main-title">🧠 Sistema RAG Inteligente</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Generación Aumentada por Recuperación - Tu asistente para análisis de documentos PDF</div>', unsafe_allow_html=True)

# Sidebar mejorado
with st.sidebar:
    st.markdown("""
        <h2 style="margin: 0">Configuración</h2>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔑 API Key")
    ke = st.text_input('Clave de OpenAI', type="password", label_visibility="collapsed")
    
    if ke:
        os.environ['OPENAI_API_KEY'] = ke
        st.success("✅ API Key configurada")
    else:
        st.warning("🔒 Ingresa tu API Key para comenzar")
    
    st.markdown("### 💡 Preguntas Sugeridas")
    
    suggested_questions = [
        "¿Cuál es el tema principal del documento?",
        "Haz un resumen ejecutivo",
        "¿Qué puntos clave menciona?",
        "¿Hay conclusiones importantes?",
        "¿Qué recomendaciones sugiere?"
    ]
    
    for i, question in enumerate(suggested_questions):
        if st.button(question, key=f"q_{i}", use_container_width=True):
            st.session_state.suggested_question = question
    st.markdown('</div>', unsafe_allow_html=True)

# Contenido principal
col1, col2 = st.columns([1, 1])

with col1:
    # Sección de carga de PDF
    st.markdown("### 📄 Carga tu Documento PDF")
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    pdf = st.file_uploader(
        "Arrastra y suelta tu PDF aquí",
        type="pdf",
        label_visibility="collapsed"
    )
    
    if pdf is not None:
        st.success(f"✅ {pdf.name} cargado correctamente")
        file_details = {
            "Nombre": pdf.name,
            "Tamaño": f"{pdf.size / 1024:.1f} KB",
            "Tipo": pdf.type
        }
        
        for key, value in file_details.items():
            st.markdown(f"**{key}:** {value}")
    else:
        st.info("📁 Selecciona un archivo PDF para comenzar el análisis")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mostrar imagen
    try:
        image = Image.open('cutie.png')
        st.image(image, width=300)
    except Exception as e:
        st.info("🖼️ Imagen de asistente no disponible")

with col2:
    # Procesar PDF y preguntas
    if pdf is not None and ke:
        try:
            # Mostrar progreso
            with st.spinner("🔍 Procesando documento..."):
                # Extraer texto del PDF
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                
                # Métricas del documento
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                with col_meta1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div>📄 Páginas</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #a855f7;">
                            {len(pdf_reader.pages)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_meta2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div>📝 Caracteres</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #c084fc;">
                            {len(text):,}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_meta3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div>🔤 Fragmentos</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #e879f9;">
                            {len(text) // 500 + 1}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Dividir texto en chunks
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                
                # Crear embeddings y base de conocimiento
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embeddings)
                
                st.success(f"🎯 Documento procesado: {len(chunks)} fragmentos listos")

            # Interfaz de preguntas
            st.markdown("### 💬 Haz tu Pregunta")
            
            # Usar pregunta sugerida si está disponible
            if 'suggested_question' in st.session_state:
                default_question = st.session_state.suggested_question
                del st.session_state.suggested_question
            else:
                default_question = ""
            
            user_question = st.text_area(
                "Escribe tu pregunta sobre el documento:",
                value=default_question,
                height=100,
                placeholder="Ej: ¿Cuál es el resumen ejecutivo de este documento?..."
            )
            
            # Inicializar historial de chat
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            if user_question:
                # Mostrar pregunta del usuario
                st.markdown(f'<div class="user-question">{user_question}</div>', unsafe_allow_html=True)
                
                with st.spinner("🤔 Analizando documento..."):
                    # Buscar documentos similares
                    docs = knowledge_base.similarity_search(user_question, k=3)
                    
                    # Configurar modelo
                    llm = OpenAI(temperature=0.3, model_name="gpt-4")
                    
                    # Cargar cadena de QA
                    chain = load_qa_chain(llm, chain_type="stuff")
                    
                    # Ejecutar cadena
                    response = chain.run(input_documents=docs, question=user_question)
                    
                    # Mostrar respuesta
                    st.markdown(f'<div class="chat-bubble">{response}</div>', unsafe_allow_html=True)
                    
                    # Guardar en historial
                    st.session_state.chat_history.append({
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'question': user_question,
                        'answer': response
                    })
            
            # Mostrar historial de chat
            if st.session_state.chat_history:
                st.markdown("### 📜 Historial de Conversación")
                for chat in reversed(st.session_state.chat_history[-5:]):
                    with st.container():
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 12px; margin: 0.5rem 0;">
                            <div style="font-size: 0.8rem; color: #c084fc;">🕒 {chat['timestamp']}</div>
                            <div style="font-weight: 600; margin: 0.3rem 0;">👤 {chat['question']}</div>
                            <div style="color: #d8b4fe;">🤖 {chat['answer'][:150]}...</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Botón para limpiar historial
                if st.button("🗑️ Limpiar Historial", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Error al procesar el PDF: {str(e)}")
            
    elif pdf is not None and not ke:
        st.warning("🔑 Por favor ingresa tu clave de API de OpenAI para continuar")
    else:
        # Estado inicial
        st.markdown("""
        <div style="text-align: center; padding: 3rem; color: #d8b4fe;">
            <h2>🚀 Comienza tu Análisis</h2>
            <p>1. Ingresa tu API Key en la barra lateral</p>
            <p>2. Carga un documento PDF</p>
            <p>3. Haz preguntas sobre el contenido</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #c084fc; padding: 2rem;">
    <p>Desarrollado con ❤️ usando Streamlit, LangChain y OpenAI</p>
    <p style="font-size: 0.8rem; color: #a855f7;">Sistema RAG - Generación Aumentada por Recuperación</p>
</div>
""", unsafe_allow_html=True)
