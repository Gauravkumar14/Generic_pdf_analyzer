import streamlit as st
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
)


# st.markdown("""
#     <style>
#     .stApp {
#         background-color: #0E1117;
#         color: #FFFFFF;
#     }
    
#     /* Chat Input Styling */
#     .stChatInput input {
#         background-color: #1E1E1E !important;
#         color: #FFFFFF !important;
#         border: 1px solid #3A3A3A !important;
#     }
    
#     /* User Message Styling */
#     .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
#         background-color: #1E1E1E !important;
#         border: 1px solid #3A3A3A !important;
#         color: #E0E0E0 !important;
#         border-radius: 10px;
#         padding: 15px;
#         margin: 10px 0;
#     }
    
#     /* Assistant Message Styling */
#     .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
#         background-color: #2A2A2A !important;
#         border: 1px solid #404040 !important;
#         color: #F0F0F0 !important;
#         border-radius: 10px;
#         padding: 15px;
#         margin: 10px 0;
#     }
    
#     /* Avatar Styling */
#     .stChatMessage .avatar {
#         background-color: #00FFAA !important;
#         color: #000000 !important;
#     }
    
#     /* Text Color Fix */
#     .stChatMessage p, .stChatMessage div {
#         color: #FFFFFF !important;
#     }
    
#     .stFileUploader {
#         background-color: #1E1E1E;
#         border: 1px solid #3A3A3A;
#         border-radius: 5px;
#         padding: 15px;
#     }
    
#     h1, h2, h3 {
#         color: #00FFAA !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Global Background & Font */
    .stApp {
        background-color: #101418;
        color: #E0E0E0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    .stSidebar {
        background-color: #181C20 !important;
        color: #D1D5DB !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #D1D5DB !important; /* Soft white-grey */
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1F2328 !important;
        color: #E0E0E0 !important;
        border: 1px solid #31363B !important;
        border-radius: 6px !important;
        padding: 10px 18px !important;
        transition: all 0.2s ease-in-out;
    }
    
    .stButton>button:hover {
        background-color: #2B3036 !important;
        border-color: #41464D !important;
    }

    /* Chat Input */
    .stChatInput input {
        background-color: #181C20 !important;
        color: #E0E0E0 !important;
        border: 1px solid #31363B !important;
        border-radius: 6px !important;
        padding: 12px !important;
    }
    
    /* User Message */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1A1E22 !important;
        border: 1px solid #2A2F34 !important;
        color: #D1D5DB !important;
        border-radius: 8px;
        padding: 14px;
        margin: 8px 0;
        box-shadow: none;
    }
    
    /* Assistant Message */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #22262C !important;
        border: 1px solid #2F353B !important;
        color: #E0E0E0 !important;
        border-radius: 8px;
        padding: 14px;
        margin: 8px 0;
        box-shadow: none;
    }

    /* File Uploader */
    .stFileUploader {
        background-color: #181C20 !important;
        border: 1px solid #31363B !important;
        border-radius: 6px !important;
        padding: 14px !important;
    }

    /* Selectbox & Input Fields */
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        background-color: #181C20 !important;
        color: #E0E0E0 !important;
        border: 1px solid #31363B !important;
        border-radius: 6px !important;
        padding: 10px;
    }
    
    /* Focus Effects */
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus {
        border-color: #4B5563 !important;
        box-shadow: 0px 0px 6px rgba(75, 85, 99, 0.5);
    }
    
    /* Spacing & Cleanup */
    .stMarkdown {
        margin-bottom: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)
st.title("🧠 MSFT OLS Handbook")
st.caption("🚀Welcome to the MSFT OLS Handbook! Ask me anything about the handbook and I'll do my best to help you out.")

#### Sidebar configuration  ####

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:14b", "deepseek-r1:7b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()


#### Initiate ChatOllama Engine ####
llm_engine = ChatOllama(
    model=selected_model,
    base_url="http://127.0.0.1:11434",
    temperature=0.3    
)

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions "
    "with strategic print statements for debugging. Always respond in English."
)

if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai","content": "Hello! I'm the MSFT OLS Handbook Assistant. How can I help you today?"}]

chat_container = st.container()

#### Display chat messages ####
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

#### Chat input and processing #####

user_query = st.chat_input("Type what you want to ask me...")

def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain |llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
        else:
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    ### generate AI response ###
    with st.spinner("🧠Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    ### Add AI response in  message log ###
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    ### Rerun to update chat display ###
    st.rerun()
