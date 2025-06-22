from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader


loader = TextLoader(
    file_path="../data/blackhole.txt"
)

docs = loader.load()
print(f"Loaded {len(docs)} documents from {loader.file_path} of type {type(docs)}")






split_text = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=0,
    # length_function=len
) 


# data_split = split_text.split_text(docs[0].page_content) 
data_split = split_text.split_documents(docs)
print("\n ---\n")  # Separator for clarity

print(f"Number of chunks: {len(data_split)}")
print(data_split)

