from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import os

class embedding {
  
}

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
split_str = []
for i in range(len(split_docs)):
  split_str.append(split_docs[i].page_content)


apikey = os.getenv('OPENAI_API_KEY')
#embeddings = OpenAIEmbeddings(key = apikey)
#print(embeddings)

# Embedding with SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print(split_docs[0])
sentence_embeddings = model.encode(split_str)
print(sentence_embeddings)

db = FAISS.from_documents(split_docs, sentence_embeddings)
query = "衣服的袖长是什么"
similar_docs = db.similarity_search(query)

# Create prompt
prompt = f"已知信息：\n{similar_docs[0].page_content}\n{similar_docs[1].page_content}请根据这些信息回答问题：\n{query}"

if __name__ == "__main__":
  print(split_docs[0])
  print(split_docs[1])
  print(similar_docs[0])
  print(prompt)
