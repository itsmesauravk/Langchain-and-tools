# from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence,RunnableParallel, RunnableLambda, RunnablePassthrough
import os

# Loading environment variables from .env file
load_dotenv()


# Creating prompt
prompt = PromptTemplate(
    template="Write a fact about {topic}",
    input_variables=["topic"]
)


# Creating model

# Use HuggingFaceHub for remote model inference
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")
    
    # Initialize HuggingFaceEndpoint LLM with explicit parameters
llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.7,  
        max_new_tokens=250,  
        repetition_penalty=1.1,  # Reduce repetition
        do_sample=True,  # Enable sampling for variety
        huggingfacehub_api_token=api_token
)


# Creating output parser
output_parser = StrOutputParser()


#pyhon function
def word_count(text):
    return len(text.split())



# CHAINS# Creating runnable chains

joke_gen_chain = RunnableSequence(prompt, llm, output_parser)

parallel_chain = RunnableParallel({
    "fact": RunnablePassthrough(),
    "word_count": RunnableLambda(word_count)
})


final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic":"Human"})

print(result)