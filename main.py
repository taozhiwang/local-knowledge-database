from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os
import openai
from langchain.evaluation.loading import load_dataset

dataset = load_dataset("question-answering-paul-graham")
loader = TextLoader("./data/paul_graham_essay.txt")

# Import text
#loader = UnstructuredFileLoader("./data/test-long.txt")

# Transform to document
data = loader.load()
print(f'documents:{len(data)}')

# Initialize tex spilitter
text_splitter = CharacterTextSplitter(chunk_size=40, chunk_overlap=5)
# Split the document
split_docs = text_splitter.split_documents(data)
docs = split_docs
print("split_docs size:",len(split_docs))

# Embedding with hugging face hub
embeddings = HuggingFaceEmbeddings()

# Store and search by using FAISS
db = FAISS.from_documents(split_docs, embeddings)
query = dataset[0]["question"]
print(query)
similar_docs = db.similarity_search(query)

# Create prompt
system_prompt = "You are a person who answers questions for people based on specified information\n"
similar_prompt = similar_docs[0].page_content + "\n" + similar_docs[1].page_content + "\n" + similar_docs[2].page_content + "\n"
question_prompt = f"Here is the question: {query}\nPlease provide an answer only related to the question and do not include any information more than that.\n"
prompt = system_prompt + "Here is some information given to you:\n" + similar_prompt + question_prompt

# Openai api setting
openai.api_key = os.environ["OPENAI_API_KEY"]

# Create response from gpt-3.5
response = openai.ChatCompletion.create(
  model = "gpt-3.5-turbo",
  temperature =  0.7,
  messages=[
        {"role": "system", "content": "你是一个根据提供的材料作出回答的助手"},
        {"role": "user", "content": prompt},
    ],
  max_tokens = 40
)

if __name__ == "__main__":
  print(split_docs[0])
  print(split_docs[1])
  print(similar_docs[0])
  print(prompt)
  print(response)
  print(response['choices'][0]['message']['content'].replace(' .', '.').strip())