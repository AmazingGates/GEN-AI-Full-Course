# In this section we will start the practical implementation demo of Vector Databases.

from langchain_community.document_loaders import PyPDFDirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import openai
from langchain_community.vectorstores import Pinecone
from langchain.chains import retrieval_qa
from langchain.prompts import PromptTemplate
import os
import pinecone
import sys


PyPDFDirectoryLoader ("pdfs")

loader = PyPDFDirectoryLoader ("pdfs")

loader.load()

data = loader.load()

data[0]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 0)

text_chunks = text_splitter.split_documents(data)

text_chunks

text_chunks[0]

print(text_chunks[0])

print(text_chunks[1].page_content)

print(text_chunks[2].page_content)

print(text_chunks[3].page_content)

print(len(text_chunks))

os.environ["OPENAI_API_KEY"] = "sk-proj-NFj08EyWI8laQ9spJBafT3BlbkFJKzklOVqO2yHxrcNMo4lT"

embedding = OpenAIEmbeddings()

embedding.embed_query("How Are You")

print(embedding.embed_query("How Are You"))

print(len(embedding.embed_query("How Are You")))

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "c1489634-2122-4dc3-a641-e16b3399b247")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV", "gcp-starter")


pinecone.init(
    api_key = PINECONE_API_KEY,
    environment = PINECONE_API_ENV
)

index_name = "test"

index = Pinecone.Index("testing")

docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embedding, index_name=index_name)

docsearch

query = "A Game Of Thrones Book 1 outperforms which models"

docs = docsearch.similarity_search(query)

docs

llm = openai()

qa = retrieval_qa.from_chain_type(llm = llm, chain_type = "stuff", retriever = docsearch.as_retriever())

query = "A Game Of Thrones Book 1 outperforms which models"

qa.run(query)

while True:
    user_input = input(f"Input Prompt: ")
    if user_input == "exit":
        print("Exiting")
        sys.exit()
    if user_input == "":
        continue
    result = qa({"query": user_input})
    print(f"Answer: {result['result']}")

