from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence
import os

from langchain_community.document_loaders import WebBaseLoader

load_dotenv()


url = "https://saurav-karki.com.np/"

loader = WebBaseLoader(web_path=url)
docs = loader.load()

print(f"Loaded {len(docs)} documents from {loader.web_path} of type {type(docs)}")
print("\n ---\n")  # Separator for clarity


prompt = PromptTemplate(
    template="Give me the summary of the following document: {document}",
    input_variables=["document"]
)

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.7,
    max_new_tokens=250,
    repetition_penalty=1.1,
    do_sample=True,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

result = chain.invoke({"document": docs[0].page_content})

print("Summary of the document:")
print("\n---\n")  # Separator for clarity
print(result)

