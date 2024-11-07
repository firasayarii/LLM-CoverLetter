import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import SecretStr
import requests

# Custom class extending SecretStr
class MySecretStr(SecretStr):
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        schema = handler(core_schema)
        return schema

# Streamlit page configuration
st.set_page_config(page_title="Cover Letter", page_icon="📧")
st.title("📧 Motivation Letter for Job Offer")

# Load the API key from Streamlit secrets
try:
    Token = st.secrets['GROQ_API_KEY']
except KeyError:
    st.error("Missing GROQ_API_KEY in Streamlit secrets.")
    st.stop()

# Initialize the ChatGroq model
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key=Token
)

# User inputs
url_input = st.text_input("Enter the URL of the job offer:")
mentions = st.text_input(
    "What do you want to mention (Personal information and your interest):",
    placeholder="I am X, currently studying in Y university, and I am interested in..."
)

# Button for submission
submit_button = st.button("Submit")

if submit_button:
    if url_input:
        try:
            # Load the web page data
            loader = WebBaseLoader(url_input)
            documents = loader.load()

            if not documents:
                st.error("Failed to load content from the provided URL.")
                st.stop()

            page_data = documents[0].page_content  # Load first document's content

            # Create the prompt template
            prompt_extract = PromptTemplate.from_template(
                """
                ### SCRAPED TEXT FROM WEBSITE:
                {page_data}
                
                ### INSTRUCTION:
                Return only a motivation letter according to this text and begin by presenting that {mentions}
                """
            )

            # Execute the prompt chain
            chain_extract = prompt_extract | llm
            res = chain_extract.invoke(
                input={
                    'page_data': page_data,
                    'mentions': mentions
                }
            )

            # Display the response
            st.text_area("Generated Motivation Letter", res.content, height=400)

        except requests.exceptions.MissingSchema:
            st.error("Please provide a valid URL (e.g., http://example.com).")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please enter a URL.")
