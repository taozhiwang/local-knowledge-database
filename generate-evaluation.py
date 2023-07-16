from langchain.evaluation.loading import load_dataset
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import openai
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.evaluation.qa import QAEvalChain
from collections import Counter
import os

def evaluation():
    dataset = load_dataset("question-answering-paul-graham")
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings)

    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Update query
    for i in range(len(dataset)):
        query = dataset[i]["question"]
        similar_docs = db.similarity_search(query)
        # Create prompt
        system_prompt = "You are a person who answers questions for people based on specified information\n"
        similar_prompt = similar_docs[0].page_content + "\n" + similar_docs[1].page_content + "\n" + similar_docs[2].page_content + "\n"
        question_prompt = f"Here is the question: {query}\nPlease provide an answer only related to the question and do not include any information more than that.\n"
        prompt = system_prompt + "Here is some information given to you:\n" + similar_prompt + question_prompt
        
        dataset[0]["question"] = prompt

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
    for i, prediction in enumerate(predictions):
        prediction["grade"] = graded_outputs[i]["text"]
    
    print(Counter([pred["grade"] for pred in predictions]))

if __name__ == "__main__":
    evaluation()
    #print(response['choices'][0]['message']['content'].replace(' .', '.').strip())