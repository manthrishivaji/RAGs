# app.py
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import pandas as pd
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_community.llms import HuggingFaceEndpoint
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# Check if the Hugging Face Hub API token is set
HUGGING_FACE_HUB_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
if not HUGGING_FACE_HUB_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file.")

# Global variables for the vector store and RAG chain
global db
global rag_chain

def process_pdf(pdf_file):
    """
    Process a PDF file to extract text, split it into chunks, and create embeddings.
    
    Args:
        pdf_file (str): Path to the PDF file.
    
    Returns:
        str: A message indicating the process is complete.
    """
    # Extract text from PDF
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_text(text=text)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    # Initialize LLM
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3", 
        max_length=128, 
        temperature=0.5
    )
    
    # Initialize Hypothetical Document Embedder
    hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
        llm,
        embeddings,
        prompt_key="web_search"
    )
    
    # Create vector store
    vectorstore = Chroma.from_texts(
        texts=docs,
        collection_name='collection-1',
        embedding=hyde_embeddings
    )
    
    # Initialize retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 2}
    )

    # Define prompt template
    template = """Answer the following question based on this context, if it doesn't exist say you don't know it:
    {context}

    Question: {question}"
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Format documents for the chain
    def format_docs(pdf_reader):
        return "\n\n".join(doc.page_content for doc in pdf_reader)
        
    # Initialize RAG chain
    global rag_chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return "Ask your Query"

def query_(query, history):
    """
    Query the RAG chain with a user question.
    
    Args:
        query (str): The user's question.
        history (list): The chat history.
    
    Returns:
        str: The response from the RAG chain.
    """
    global rag_chain
    response = rag_chain.invoke(query)
    return response

# Create Gradio interface
with gr.Blocks() as combined_interface:
    with gr.Tab("File Upload Interface"):
        iface1 = gr.Interface(
            fn=process_pdf, 
            inputs="file",
            outputs="text"
        )
    with gr.Tab("Chat Interface"):
        iface2 = gr.ChatInterface(query_)

# Launch the interface
combined_interface.launch()