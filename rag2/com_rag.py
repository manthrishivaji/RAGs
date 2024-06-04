# perfect
# This code working perfectly fine wihtout wrong answer for unknown query .
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
import getpass
import os
import gradio as gr

if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Provide your HUGGINGFACEHUB TOKEN")


global db
global rag_chain

def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1000,
                chunk_overlap = 200,
                length_function = len
            )
    docs = text_splitter.split_text(text=text)
    
    # Embedding
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    #llm
    llm=HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.1, max_length=512)
    
    #hyde
    hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(llm,
                                                    embeddings,
                                                    prompt_key = "web_search")
    
    #vectorstore
    vectorstore = Chroma.from_texts(texts=docs,
                                    collection_name='collection-1',
                                    embedding=hyde_embeddings)
    #retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    #template
    template = """Answer the following question based on this context:
    {context}

    Question: {question}+"in the document"
    """
    #prompt
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(pdf_reader):
        return "\n\n".join(doc.page_content for doc in pdf_reader)
        
    #chain
    global rag_chain
    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    return "Ask your Query"

def query_(query,history):
    global rag_chain
    response = rag_chain.invoke(query)
    return response

with gr.Blocks() as combined_interface:
    with gr.Tab("fileupload interface"):
        iface1 = gr.Interface(fn=process_pdf, inputs="file",outputs="text")
    with gr.Tab("Chatinterface"):
        iface2 = gr.ChatInterface(query_
            # chatbot=gr.Chatbot(height=300),
            # textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
            # title="Yes Man",
            # description="Ask Yes Man any question",
            # theme="soft",
            # examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
            # cache_examples=True,
            # retry_btn=None,
            # undo_btn="Delete Previous",
            # clear_btn="Clear",
        )
combined_interface.launch() 


