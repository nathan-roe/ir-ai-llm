import time
import whisper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

print("Started running...")
transcribe_start_time = time.time()
model = whisper.load_model("medium.en")
result = model.transcribe("football.m4a", fp16=False)
transcribe_end_time = time.time()
print(f'Total processing time {transcribe_end_time - transcribe_start_time}')

rag_start_time = time.time()
transcipt = result["text"]
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = splitter.split_text(transcipt)

embeddings = OllamaEmbeddings()
dosearch = FAISS.from_texts(texts, embeddings, metadatas=[{"source": str(text)} for text in range(len(texts))])

llm = Ollama(model='llama2')
rag_prompt = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You summarize the contents of a transcribed football broadcast.
                Briefly summarize the provided audio file transcription.
                Ensure your summary matches the official 2023 NFL rule book, but don't include the section number in your response.
                Do not include information other than the summary in your response.
                Keep your response to three sentences or less.
                Use the following question as the basis for your response.
                \nQuestion: {question}
                \nFootball Broadcast: {context}
                \nAdditional Context: This is a game between the Los Angeles Charges and the San Francisco 49ers
                """
            )
        )
    ]
)
chain = load_qa_chain(llm, chain_type="stuff", prompt=rag_prompt)


query = "What is the summary of this broadcast?"
docs = dosearch.similarity_search(query)

response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
rag_end_time = time.time()

print("response: ", "".join(response["output_text"]))
print(f'Total RAG time {rag_end_time - rag_start_time}')
