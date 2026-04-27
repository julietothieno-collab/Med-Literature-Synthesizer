import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

st.set_page_config(page_title="Med-Synth AI", page_icon="🧬", layout="wide")

# Custom Medical CSS
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .stButton>button { background-color: #1B4F72; color: white; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Medical Literature Synthesizer")
st.markdown("---")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Google Gemini API Key", type="password", help="Get a free key at aistudio.google.com")
    st.info("This app uses Google Gemini Pro to synthesize findings across multiple medical papers.")
    pdf_docs = st.file_uploader("Upload Research Papers (PDF)", accept_multiple_files=True)
    
    if st.button("Analyze Literature"):
        if api_key and pdf_docs:
            with st.spinner("Decoding medical text..."):
                os.environ["GOOGLE_API_KEY"] = api_key
                # 1. Extract Text
                raw_text = ""
                for pdf in pdf_docs:
                    reader = PdfReader(pdf)
                    for page in reader.pages:
                        raw_text += page.extract_text() or ""
                
                # 2. Chunk text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
                chunks = text_splitter.split_text(raw_text)
                
                # 3. Create Vector Store
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.from_texts(chunks, embeddings)
                st.session_state.vector_store = vector_store
                st.success("Analysis Complete!")
        else:
            st.error("Please provide an API key and upload PDFs.")

# UI for user query
user_question = st.text_input("Enter your research question:")

if user_question and "vector_store" in st.session_state:
    docs = st.session_state.vector_store.similarity_search(user_question)
    
    # Advanced Medical Research Prompt
    prompt_template = """
    You are an expert Medical Research Assistant. Synthesize an answer to the clinical question using ONLY the provided evidence.
    Format your response with:
    1. A Summary of findings.
    2. Comparison/Conflicts between the studies (if any).
    3. Methodological strengths/weaknesses noted.
    
    Context: {context}
    Question: {question}
    
    Evidence-Based Synthesis:
    """
    
    os.environ["GOOGLE_API_KEY"] = api_key
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.markdown("### 📊 Evidence Synthesis Report")
    st.write(response["output_text"])
