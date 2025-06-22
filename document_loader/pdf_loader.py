from langchain_community.document_loaders import PyPDFLoader


"""
PyPDFLoader     = Load PDF files that are mostly text-based.
PDFPlumberLoader = Load PDF files that are mostly table, column based.
UnstructuredPDFLoader ,  AmazonTextractPDFLoader = Load PDF files that are mostly scanned images.

"""

loader = PyPDFLoader(
    file_path="../data/the annoted transformer.pdf"
)

docs = loader.load()

print(f"Loaded {len(docs)} documents from {loader.file_path} of type {type(docs)}")

print("Document content:")

# for doc in docs:
#     print(doc.page_content)
#     print("\n---\n")  # Separator between documents

print(docs[0])