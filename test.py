# Import necessary libraries

try:
    import openai
    import awadb
except ImportError as exc:
    raise ImportError(
        "Could not import libraries. "
        "Please install it with `pip install awadb` or `pip install openai`"
    ) from exc

import os

    
# Load the data file
from langchain.document_loaders import TextLoader
loader = TextLoader("state_of_the_union.txt")

# Transform to document
data = loader.load()
print(f'documents:{len(data)}')

# Initialize tex spilitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=40, chunk_overlap=5)
# Split the document
split_docs = text_splitter.split_documents(data)
print("split_docs size:",len(split_docs))

# Save the embedded texts by Awadb
"""
from langchain.vectorstores import AwaDB
db = AwaDB.from_documents(split_docs)

# Set the question
query = "What were the two main things the author worked on before college?"
# Similarity search results
similar_docs = db.similarity_search(query)
print(similar_docs)
"""

texts = [text.page_content for text in split_docs]

print(texts)

awadb_client = awadb.Client()
awadb_client.Create("testdb1")


for text in texts:
    print([text])
    awadb_client.Add([text])

# Set the question
query = "What were the two main things the author worked on before college?"
awadb_client.AddTexts
# Similarity search results
similar_docs = awadb_client.Search(query, 3)

#for i in range(len(split_docs)-1):
#    awadb_client.Add([for text in ])

# 3. Add docs to the table. Can also update and delete the doc!
#for text in split_docs:
#    awadb_client.Add([text.page_content])