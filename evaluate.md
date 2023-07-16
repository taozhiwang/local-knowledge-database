# Evaluate embedding models based on performance on question answering tasks

## Background

The traditional way of Embedding evaluation includes the use of Spearman rank coefficient or Pearson coefficient to evaluate the similarity of embedding. 

In addition, there are also ways to evaluate the semantic similarity of two words that are close to each other in space using lexical similarity datasets such as wordsim353. We can also examine the analogy, which is whether the model can establish a linear relationship between words' embeddings.

However, these evaluation approaches do not well characterize the performance for embedding model when they were used for question answering. 

When we embedded a piece of text and store it in the database, we want a well-performing model embedded another semantically close sentence, the question asked by the user, into a similarly distant result. This way we can retrieve the precise content as part of the prompt and have the more suitable answer given by the large language models.

## Evaluation

Therefore, we will test the performance of different embedding models on question answering through several datasets and a series of questions related to it. Here is the code for evaluation.

### HuggingFaceEmbeddings

We first test the performance of  `sentence-transformers/all-mpnet-base-v2` from **HuggingFaceEmbeddings**

#### Save Embeddings

```python
from langchain.evaluation.loading import load_dataset
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
import openai
from langchain.vectorstores import FAISS
import pandas as pd
import os

# Load the dataset which contains questions and answers
dataset = load_dataset("question-answering-paul-graham")

# Load the essay
loader = TextLoader("./data/paul_graham_essay.txt")
data = loader.load()

# Create split file
text_splitter = CharacterTextSplitter(chunk_size=40, chunk_overlap=5)
split_docs = text_splitter.split_documents(data)

# Embedding and save it locally
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(split_docs, embeddings)
db.save_local("faiss_index")
```

After embedding, we can find a folder called `faiss-index` under the current working direcotry.

#### Evaluatoin

```python
from langchain.evaluation.loading import load_dataset
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import openai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.evaluation.qa import QAEvalChain
from collections import Counter
import os
```

Here we load the data set and then load the embeddings through **FAISS**


```python
dataset = load_dataset("question-answering-paul-graham")
embeddings = HuggingFaceEmbeddings()
db = FAISS.load_local("faiss_index", embeddings)

# Obtain openai api key through environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]
```

```bash
Found cached dataset json 
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 53.35it/s]
```

Then we need to create a pipline for question answering and apply it to all questions

```python
chain = RetrievalQA.from_chain_type(
    llm=OpenAI(), # Use openai as output large language model
    chain_type="stuff",
    retriever=db.as_retriever(),
    input_key="question",
)
# Apply the chain to all questions in dataset
predictions = chain.apply(dataset)

# Evaluation
llm = OpenAI(temperature=0.7)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(
    dataset, predictions, question_key="question", prediction_key="result"
)
```


```python
for i, prediction in enumerate(predictions):
    prediction["grade"] = graded_outputs[i]["text"]

print(Counter([pred["grade"] for pred in predictions]))
```

We can view the result of this test through counter. 

We can see that by utilizing the embedding from `all-mpnet-base-v2`, our model answer correctly on 21 questions and incorrect in 1 question.

    Counter({' CORRECT': 21, ' INCORRECT': 1})

```python
incorrect = [pred for pred in predictions if pred["grade"] == " INCORRECT"]
incorrect[0]
```

We also tested the performance of this model in another dataset `state of the unioin`. You can gain this dataset by using the following code.

```python
dataset = load_dataset("question-answering-state-of-the-union")
```

Here is the result of this test.

```
Counter({' CORRECT': 9, ' INCORRECT': 2})

# Here are two incorrect questions we received.
{'answer': 'The four common sense steps suggested by the author to move forward safely are: stay protected with vaccines and treatments, prepare for new variants, end the shutdown of schools and businesses, and stay vigilant.', 'question': 'What are the four common sense steps that the author suggests to move forward safely?', 'result': " I don't know.", 'grade': ' INCORRECT'}
{'answer': 'The Unity Agenda for the Nation includes four big things that can be done together: beat the opioid epidemic, take on mental health, support veterans, and strengthen the Violence Against Women Act.', 'question': 'What is the Unity Agenda for the Nation that the President is offering?', 'result': ' The Unity Agenda for the Nation that the President is offering is to build a national network of 500,000 electric vehicle charging stations, to begin to replace poisonous lead pipes, to provide affordable high-speed internet for every American, and to ensure clean water for all Americans.', 'grade': ' INCORRECT'}
```

### OpenAIEmbedding

Now we will test the performace of OpenAiEmbedding in question answering.

Use `from langchain.llms import OpenAI` to import OpenAi library

#### Save Embedding

The process of saving embedding is similar to hat we did before. Noticed that we need to set `openai.api_key` for embeddings

```python
openai.api_key = os.environ["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(split_docs, embeddings)
db.save_local("openai_index")
```

The result for OpenAIEmbedding is below

```bash
Counter({' CORRECT': 22})
```

The result of `OpenAIEmbedding` on `State of the union` dataset is here.

```
Counter({' CORRECT': 9, ' INCORRECT': 2})

{'answer': 'The four common sense steps suggested by the author to move forward safely are: stay protected with vaccines and treatments, prepare for new variants, end the shutdown of schools and businesses, and stay vigilant.', 'question': 'What are the four common sense steps that the author suggests to move forward safely?', 'result': " I don't know.", 'grade': ' INCORRECT'}
{'answer': 'The Unity Agenda for the Nation includes four big things that can be done together: beat the opioid epidemic, take on mental health, support veterans, and strengthen the Violence Against Women Act.', 'question': 'What is the Unity Agenda for the Nation that the President is offering?', 'result': " I don't know.", 'grade': ' INCORRECT'}
```



## Results

We show in this passage a way to embedding the same textual content by using different embedding models and subsequently evaluating these models based on their performance under the same problem set. 

The results show that the embedding model performs well for this type of dataset and has a good correctness rate on the question responses. Specifically, after testing, OpenAIEmbedding performs even better, but not much different from other models

Specifically, OpenAIEmbedding's combined correctness on both datasets is 93.93%. and the sentiment- transformer's combined correctness is 90.90%.

