from langchain.text_splitter import RecursiveCharacterTextSplitter, Language


text = """
class Animal:
    __init__(self, name):
        self.name = name
    def speak(self):
        return f"{self.name} makes a sound."
class Dog(Animal):
    def speak(self):
        return f"{self.name} barks."
class Cat(Animal):
    def speak(self):
        return f"{self.name} meows."
"""


splitter = RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size=100,
    chunk_overlap=10,
) 

data_split = splitter.split_text(text)

print(f"Number of chunks: {len(data_split)}")
print(data_split[0])
