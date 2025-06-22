from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


"""
DirectoryLoader = Load all files in a directory, recursively.
**/*.txt = Load all text files in a directory, recursively.
**/*.pdf = Load all PDF files in a directory, recursively.
**/*.csv = Load all CSV files in a directory, recursively.
**/*.json = Load all JSON files in a directory, recursively.
**/* = Load all files in a directory, recursively.

** = recursive search through subfolders
"""

loader = DirectoryLoader(
    path="../data",
    glob="**/*.pdf", 
    loader_cls=PyPDFLoader,
    show_progress=True,
)

# docs = loader.load()          # Load all data at once (not recommended for large datasets)
docs = loader.lazy_load()  # Use lazy_load to load data in chunks

"""
loader.load()    => load function loads all the data in the memory, if the data is too large, it will cause memory error.
loader.lazy_load() => lazy_load function loads the data in chunks, it is useful for large data.
"""

# print(f"Loaded {len(docs)} documents from {loader.path} of type {type(docs)}")

# print("Document content:")

# print(docs[8])  # Print the first document's content

for doc in docs:
    print(doc.metadata)
    print("\n---\n")  # Separator between documents

