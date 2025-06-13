# At the VERY TOP of app.py
import sys

# This is the "magic" fix for the sqlite3 issue on Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import yaml
import os
import re # Import the regular expression module
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. CONFIGURATION ---
RAG_DATABASE = "data/cannondale_chroma_db"
EMBEDDING_MODEL = 'text-embedding-3-small'
CHAT_LLM_OPTIONS = ["gpt-4o-mini", "gpt-4o"]

# --- 2. PAGE SETUP ---
st.set_page_config(page_title="Cannondale AI Bike Assistant", page_icon="🚲")
st.title("🚲 Cannondale AI Bike Assistant")

# --- 3. API KEY & SESSION STATE ---
# Securely set the API key from Streamlit's secrets management
try:
    os.environ['OPENAI_API_KEY'] = st.secrets['openai']
except KeyError:
    st.error("OpenAI API key not found. Please add it to your Streamlit secrets.")
    st.stop()

# Set up chat history
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello! How can I help you find the perfect Cannondale bike today?")

# --- 4. RAG CHAIN CREATION ---
@st.cache_resource(show_spinner="Loading RAG chain...")
def create_rag_chain():
    """
    Creates and caches the LangChain RAG (Retrieval-Augmented Generation) chain.
    """
    # Initialize the vector store and retriever
    embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=RAG_DATABASE, embedding_function=embedding_function)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # Initialize the Chat LLM
    llm = ChatOpenAI(model=st.session_state.LLM, temperature=0.7)

    # --- Chain Step 1: Contextualize Question ---
    # This part reformulates the user's latest question to be a standalone question
    # based on the chat history.
    contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # --- Chain Step 2: Answer Question with Context ---
    # This is the core RAG step where the LLM answers based on retrieved documents.

    # **MODIFICATION 1: Define a prompt to format each retrieved document.**
    # This is the critical change. It tells LangChain how to structure the data from
    # each document before adding it to the final context. We explicitly include the metadata fields.
    document_prompt = PromptTemplate.from_template(
        """---
Bike Name: {name}
Image URL: {image_url}
Source: {url}

Content: 
{page_content}
---"""
    )

    # **MODIFICATION 2: Update the final QA system prompt.**
    # This prompt now instructs the LLM to look for the "Image URL" field in the
    # context and use a specific tag format if it finds one.
    qa_system_prompt = """You are an expert assistant for the Cannondale bike company. Use the following pieces of retrieved context to answer the question.
    Your primary goal is to be helpful and provide accurate information about the bikes.
    
    If the context for a bike includes an "Image URL", you MUST embed that exact URL in your response using this format: [IMAGE: URL_FROM_CONTEXT]
    
    If the information is not in the context, say you don't have enough information. Be concise and helpful.

    Context:
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    # **MODIFICATION 3: Create the final document combination chain.**
    # We pass the `document_prompt` here to ensure our metadata is included.
    question_answer_chain = create_stuff_documents_chain(
        llm, 
        qa_prompt,
        document_prompt=document_prompt  # This activates the custom document formatting
    )
    
    # Combine the history-aware retriever and the QA chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# --- 5. RESPONSE RENDERING ---
def display_response_with_images(response_text):
    """
    Parses the LLM response to find image tags and renders them with Streamlit.
    """
    # Regex to find all occurrences of [IMAGE: URL]
    image_pattern = r"\[IMAGE: (.*?)\]"
    
    # Split the response text by the image tags
    parts = re.split(image_pattern, response_text)
    
    for i, part in enumerate(parts):
        # Even-indexed parts are text, odd-indexed parts are URLs

        if i % 2 == 1:
            # This part is SUPPOSED to be a URL. Let's check.
            # If it's a valid web link, display the image.
            if part.strip().startswith(('http://', 'https://')):
                with st.spinner("Loading image..."):
                    st.image(part.strip(), use_container_width=True)
            # Otherwise, it's just text that got caught by mistake. Print it.
            else:
                if part:
                    st.write(f"_[Image: {part}]_") # Display it as text instead
        
        
        
        
        else:
            # This is plain text, display it if it's not empty
            if part:
                st.write(part)

# --- 6. CHAT INTERFACE LOGIC ---
st.sidebar.title("Configuration")
st.session_state.LLM = st.sidebar.selectbox("Choose OpenAI model", CHAT_LLM_OPTIONS, index=0)

# Display chat messages from history
for msg in msgs.messages:
    with st.chat_message(msg.type):
        # Use our custom display function for AI messages
        if msg.type == "ai":
            display_response_with_images(msg.content)
        else:
            st.write(msg.content)


# Handle new user input
if prompt := st.chat_input("Ask about a bike model or features..."):
    st.chat_message("human").write(prompt)

    # Configure the RAG chain with session history
    rag_chain_with_history = RunnableWithMessageHistory(
        create_rag_chain(),
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    # Stream the AI response
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            # Invoke the chain to get the response
            response = rag_chain_with_history.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": "any"}}
            )
            
            # Use our custom display function to render the final answer
            display_response_with_images(response["answer"])

# Sidebar for debugging
with st.sidebar.expander("View Message History"):
    st.json(st.session_state.langchain_messages)
