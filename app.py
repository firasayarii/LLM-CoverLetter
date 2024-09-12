import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from utilities import Token 
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key=Token,
    # other params...
)






st.title("📧 Motivation letter for Job offer")
url_input = st.text_input("Enter a URL:", value="https://www.welcometothejungle.com/fr/companies/quicksign/jobs/stage-en-data-science-f-h_paris?q=7367ad96b6fc11fd116aa0f388f9e8ca&o=2b3b8224-eac0-43d7-b21f-db9f8d49b09f")
mentions = st.text_input("What do you want to mention (Personal information and your interest) :")
loader = WebBaseLoader(url_input)
page_data = loader.load().pop().page_content


prompt_extract = PromptTemplate.from_template(
        """
        ### SCRAPED TEXT FROM WEBSITE:
        {page_data}
        ### INSTRUCTION:
        return only a motivation letter according to this text and begin by presnting me that {mentions}
        
            
        """
)

submit_button = st.button("Submit")
chain_extract = prompt_extract | llm 
res = chain_extract.invoke(input={'page_data':page_data,'mentions': mentions})

if submit_button:
    st.text_area("Response",res.content, height=400)
