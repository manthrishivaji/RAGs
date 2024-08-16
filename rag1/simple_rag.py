import textwrap
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain_community.llms import HuggingFaceEndpoint
import getpass
import os
import gradio as gr

if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Provide your HUGGINGFACEHUB TOKEN")

global db
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

    print(len(docs))

    # Embedding
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    print("embed")
    global db
    db = FAISS.from_texts(docs, embeddings)
    print("dq")
    return (db)


# model_cache = {}

# def load_llm(model_name):
#     global model_cache
#     if model_name in model_cache:
#         return model_cache[model_name]
#     else:
#         llm=HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#                             temperature=0.1,
#                               max_length=512)
        
#         model_cache[model_name] = llm
#         return llm

def query_pdf(query,state):
    global db
    if state is None:
        state = []

    # dq = db.similarity_search(query)
    # print(wrap_text_preserve_newlines(str(dq[0].page_content)))
    
    llm=HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                            temperature=0.1,
                              max_length=512)
    
    
    chain = load_qa_chain(llm, chain_type="stuff")
    print("chain")
    dq1 = db.similarity_search(query)
    response=chain.run(input_documents=dq1, question=query)
    # print(wrap_text_preserve_newlines(str(response)))
    return response
   

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

   
with gr.Blocks() as combined_interface:
    with gr.Tab("fileupload interface"):
        iface1 = gr.Interface(fn=process_pdf, inputs="file",outputs="text")
    with gr.Tab("Chatinterface"):
        iface2 = gr.ChatInterface(query_pdf
                                  )

combined_interface.launch()


