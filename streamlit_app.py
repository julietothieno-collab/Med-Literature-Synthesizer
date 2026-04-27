import streamlit as st
import os

# --- 1. ROBUST IMPORTS ---
try:
    from PyPDF2 import PdfReader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    # Safety settings import
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError as e:
    st.error(f"Library Error: {e}")
    st.stop()

# --- 2. PREMIUM MEDICAL UI DESIGN ---
st.set_page_config(page_title="Med-Synth AI", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    h1, h2, h3 { color: #1B4F72 !important; font-weight: 800 !important; }
    p, li, label, .stMarkdown { color: #000000 !important; font-size: 1.1rem !important; }
    section[data-testid="stSidebar"] { background-color: #F0F4F8 !important; border-right: 2px solid #1B4F72; }
    .stTextInput>div>div>input { border: 2px solid #1B4F72 !important; color: #000000 !important; }
    .stButton>button { background-color: #1B4F72 !important; color: white !important; border-radius: 10px !important; font-weight: bold !important; height: 3em !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR (SETUP) ---
with st.sidebar:
    st.title("🧰 Control Panel")
    st.markdown("---")
    api_key = st.text_input("1. Enter Gemini API Key", type="password")
    pdf_docs = st.file_uploader("2. Upload Research PDFs", accept_multiple_files=True)
    
    if st.button("3. START PROCESSING"):
        if api_key and pdf_docs:
            with st.spinner("🔍 Reading Medical Literature..."):
                text = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted: text += extracted
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_text(text)
                
                # Using local embeddings (Fixed the previous crash)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.from_texts(chunks, embeddings)
                
                st.session_state.vector_store = vector_store
                st.session_state.api_key = api_key
                st.success("✅ Analysis Complete!")
        else:
            st.warning("⚠️ Missing Key or PDFs")

# --- 4. MAIN INTERFACE ---
st.title("🧬 Medical Literature Synthesizer")
st.markdown(f"**Researcher:** Juliet Sera Othieno, MBBS Candidate")
st.markdown("---")

if "vector_store" not in st.session_state:
    st.info("👈 **Step 1:** Enter your API key and upload PDFs in the sidebar.")
else:
    st.markdown("### 📝 Ask Your Clinical Question")
    user_query = st.text_input("", placeholder="e.g. Briefly summarize the methodology and findings...", key="query_input")

    if user_query:
        with st.spinner("🔬 Synthesizing Research Findings..."):
            # Set API Key
            os.environ["GOOGLE_API_KEY"] = st.session_state.api_key
            
            # Retrieve context
            docs = st.session_state.vector_store.similarity_search(user_query, k=4)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # --- THE FIX: RELAXED SAFETY SETTINGS ---
            # This prevents the "redacted error" caused by safety filters blocking medical text
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            try:
                # Use gemini-pro for better reliability in clinical work
                llm = ChatGoogleGenerativeAI(
                    model="gemini-pro", 
                    temperature=0.1,
                    safety_settings=safety_settings
                )
                
                clinical_prompt = f"""
                You are a Professional Medical Data Analyst. 
                Answer the question based ONLY on the provided context. 
                Provide a structured summary with bullet points for data findings.
                
                CONTEXT:
                {context_text}
                
                QUESTION:
                {user_query}
                """
                
                response = llm.invoke([HumanMessage(content=clinical_prompt)])
                
                st.markdown("---")
                st.markdown("### 📊 Evidence Synthesis Report")
                st.write(response.content)
                
            except Exception as e:
                st.error(f"AI Error: {str(e)}")
                st.info("Tip: Check if your API Key is correct and has been funded/verified at Google AI Studio.")
