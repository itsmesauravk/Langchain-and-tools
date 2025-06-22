# from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel
import os

# Loading environment variables from .env file
load_dotenv()


# Creating prompt
prompt = PromptTemplate(
    template="Write a short funny tweet about topic : {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template='Write a interesting fact for linkedin - {topic}',
    input_variables=['topic']
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

# Creating runnable parallel
chain1 = RunnableSequence(
        prompt,
        llm,
        output_parser,  
        
)

chain2  = RunnableSequence(
    prompt2,
    llm,
    output_parser
)

parallel_chain = RunnableParallel({
    "tweet": chain1,
    "linkedin_fact": chain2
})

# Running the sequence with a specific topic
result = parallel_chain.invoke({"topic": "black hole"})

print(result)

