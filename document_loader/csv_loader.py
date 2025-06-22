from langchain_community.document_loaders import CSVLoader


loader = CSVLoader(file_path="../data/gov_services.csv")


docs = loader.load()

print(len(docs))
print(docs[0])

# print(docs)