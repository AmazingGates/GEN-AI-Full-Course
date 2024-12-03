import os

os.environ["OPENAI_API_KEY"] = "sk-proj-NFj08EyWI8laQ9spJBafT3BlbkFJKzklOVqO2yHxrcNMo4lT"


from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import openai
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import embeddings
from langchain.chains import RetrievalQA

loader = DirectoryLoader("C:/Users/alpha/Generative AI/pdfs/A Game Of Thrones Book 1.pdf")

loader.load()

document = loader.load()

document

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text = text_splitter.split_documents(document)

print(text)

print(text[0])

print(text[0].page_content)

len(text)


persist_directory = "db"

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents = text,
    embedding = embedding,
    persist_directory = persist_directory
)

vectordb.persist()

vectordb = None

vectordb = Chroma(persist_directory = persist_directory,
                  embedding_function = embedding) 

vectordb

retriever = vectordb.as_retriever() 

docs = retriever.get_relevant_documents("How much money did Microsoft raise")

docs

retriever = vectordb.as_retriever(search_kwargs = {"k": 2})

llm = openai()

llm

qa_chain = RetrievalQA.from_chain_type(llm = openai(),
                                       chain_type = "stuff",
                                       retriever = retriever,
                                       return_source_documents = True)

def process_llm_response(llm_response):
    print(llm_response["result"])
    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])

query = "How much money did Microsoft raise?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

llm_response

process_llm_response(llm_response)
