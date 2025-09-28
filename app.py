from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask Your PDF")
    st.header("Ask Your PDF")
    
    # uploading the file
    pdf=st.file_uploader("Upload your PDF", type="pdf")
    
    #extract the text
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        
        #split into chunks
        text_splitter=CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks=text_splitter.split_text(text)
        
        
        #creating embeddings
        embeddings=OpenAIEmbeddings()
        knowledge_base=FAISS.from_texts(chunks,embeddings)
        

if __name__=='__main__':
    main()