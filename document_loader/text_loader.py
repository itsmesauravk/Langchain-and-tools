from langchain_community.document_loaders import TextLoader


loader = TextLoader(
    file_path="../data/blackhole.txt",
    autodetect_encoding=True,
)

docs = loader.load()

print(f"Loaded {len(docs)} documents from {loader.file_path} of type {type(docs)}")
print("Document content:")
print(docs[0]) 