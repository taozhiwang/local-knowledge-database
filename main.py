from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
import openai

# Import text
loader = UnstructuredFileLoader("./data/test.txt")

# Transform to document
data = loader.load()
print(f'documents:{len(data)}')

# Initialize tex spilitter
text_splitter = CharacterTextSplitter(chunk_size=25, chunk_overlap=5)
# Split the document
split_docs = text_splitter.split_documents(data)
docs = split_docs
print("split_docs size:",len(split_docs))

#apikey = os.getenv('OPENAI_API_KEY')
#embeddings = OpenAIEmbeddings(key = apikey)
#print(embeddings)

# Embedding with hugging face hub
embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(split_docs, embeddings)
query = "衣服的袖长是什么"
similar_docs = db.similarity_search(query)

# Create prompt
prompt = f"已知信息：\n{similar_docs[0].page_content}\n{similar_docs[1].page_content}请根据这些信息回答问题：\n{query}"
 
openai.api_key = os.environ['OPENAI_API_KEY']
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "user", "content": prompt},
    ]
)

if __name__ == "__main__":
  print(split_docs[0])
  print(split_docs[1])
  print(similar_docs[0])
  print(prompt)
  print(response.choices.messages.content)