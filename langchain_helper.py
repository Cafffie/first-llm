import langchain
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFDirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import warnings
warnings.filterwarnings("ignore")


os.environ['GOOGLE_API_KEY']
genai.configure(api_key= os.environ['GOOGLE_API_KEY'])

llm = ChatGoogleGenerativeAI(model="gemini-pro",
                              temperature=0.3,
                              convert_system_message_to_human=True)
embeddings= GoogleGenerativeAIEmbeddings(model= "models/embedding-001")
vectordb_file_path= "faiss_db"

def create_vector_db():
    csv_loader= CSVLoader(file_path="cleaned_531_words.csv", source_column= "words")
    data= csv_loader.load()
    vectordb= FAISS.from_documents(documents=data, embedding= embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb= FAISS.load_local(vectordb_file_path, embeddings)
    retriever= vectordb.as_retriever()
    prompt_template= """Given the following context and a question, generate an answer based on this context.
    In the answer try to provide as much text as possible from "meaning" section in the source document.
    Give the meaning of any word given.

    CONTEXT: {context}
    QUESTION: {question} """

    prompt = PromptTemplate(
        template= prompt_template,
        input_variables= ["context", "question"]
    )
    chain= RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever= retriever,
                                        input_key= "query",
                                        return_source_documents=True,
                                        chain_type_kwargs= {"prompt": prompt}
            )
    return chain

if __name__ == "__main__":
    chain= get_qa_chain()

    print(chain("What is the meaning of house")["result"])















