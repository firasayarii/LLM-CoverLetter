import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import requests

# Load the API key from Streamlit secrets
Token = st.secrets['GROQ_API_KEY']
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key=Token,
    # other params...
)

# Streamlit page configuration
st.set_page_config(page_title="Cover Letter", page_icon="📧")
st.title("📧 Motivation Letter for Job Offer")

# User inputs
url_input = st.text_input("Enter the URL of the offer:")
mentions = st.text_input("What do you want to mention (Personal information and your interest):")

# Button for submission
submit_button = st.button("Submit")

if submit_button:
    if url_input:  # Check if URL is provided
        try:
            # Load the web page data
            loader = WebBaseLoader(url_input)
            page_data = loader.load().pop().page_content
            
            # Create the prompt template
            prompt_extract = PromptTemplate.from_template(
                """
                ### SCRAPED TEXT FROM WEBSITE:
                {page_data}
                ### INSTRUCTION:
                Return only a motivation letter according to this text and begin by presenting me that {mentions}
                """
            )
            
            # Execute the chain extract
            chain_extract = prompt_extract | llm 
            res = chain_extract.invoke(input={'page_data': page_data, 'mentions': mentions})

            # Display the response
            st.text_area("Response", res.content, height=400)
        
        except requests.exceptions.MissingSchema:
            st.error("Please make sure to provide a valid URL for the offer (e.g., http://example.com).")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a URL.")
