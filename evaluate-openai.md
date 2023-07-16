```python
from langchain.evaluation.loading import load_dataset
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
import openai
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.evaluation.qa import QAEvalChain
from collections import Counter
import pandas as pd
import numpy as np
import os
```


```python
dataset = load_dataset("question-answering-paul-graham")
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("openai_index", embeddings)

openai.api_key = os.environ["OPENAI_API_KEY"]
```

    Found cached dataset json (/Users/taozhiwang/.cache/huggingface/datasets/LangChainDatasets___json/LangChainDatasets--question-answering-paul-graham-76e8f711e038d742/0.0.0/8bb11242116d547c741b2e8a1f18598ffdd40a1d4f2a2872c7a28b697434bc96)



      0%|          | 0/1 [00:00<?, ?it/s]



```python
chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=db.as_retriever(),
    input_key="question",
)
predictions = chain.apply(dataset)

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

    Counter({' CORRECT': 22})



```python

```
