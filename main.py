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

# Embedding with hugging face hub
embeddings = HuggingFaceEmbeddings()

# Store and search by using FAISS
db = FAISS.from_documents(split_docs, embeddings)
query = "这款衣服的袖型是什么"
similar_docs = db.similarity_search(query)

# Create prompt
prompt = f"已知信息：\n{similar_docs[0].page_content}\n{similar_docs[1].page_content}\n请根据这些信息回答问题:\n{query}\n请仅对上述问题作出回答, 不要包含无关信息\n"

# Openai api setting
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
print(openai.api_base)
deployment_name = "Breadcrumbsautomatic-assessment-generation"
openai.api_version = "2023-05-15" # this may change in the future
openai.api_type = "azure"
#response = openai.Completion.create(model="gpt-3.5-turbo",engine=deployment_name, prompt=prompt, max_tokens=20)

# Create response from gpt-3.5
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  engine=deployment_name,
  temperature =  0.7,
  messages=[
        {"role": "system", "content": "你是一个根据提供的材料作出回答的助手"},
        {"role": "user", "content": prompt},
    ],
  max_tokens=20
)


if __name__ == "__main__":
  print(split_docs[0])
  print(split_docs[1])
  print(similar_docs[0])
  print(prompt)
  print(response)
  print(response['choices'][0]['message']['content'].replace(' .', '.').strip())