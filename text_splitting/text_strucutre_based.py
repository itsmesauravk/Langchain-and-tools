from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
)


data = """
This is a long text that needs to be split into smaller chunks. The text contains multiple sentences and paragraphs, and we want to ensure that each chunk is coherent and meaningful. The goal is to create chunks that are easy to process and understand, while still retaining the context of the original text.
This is another paragraph that continues the discussion. It provides additional information and context, which is important for understanding the overall message of the text. Each chunk should ideally contain complete thoughts or ideas, making it easier for further processing or analysis.
This is a third paragraph that adds even more depth to the content. It is crucial to maintain the flow of information, so that when the text is split, each part still makes sense on its own. The splitting process should not disrupt the narrative or logical progression of the text.
"""


chunks = splitter.split_text(data)

print(f"Number of chunks: {len(chunks)}")

print(chunks)


