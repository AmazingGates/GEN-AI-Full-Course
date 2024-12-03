# In this section we will be going over the Chroma database and noting the differences between this DB and the
#Pinecone DB.

# Chroma is a database for building AI applications with embeddings. It comes with everything we need to
#get started built-in., and runs on our local machine. 

# Chroma is the open-source embedding database. Chroma makes it easy to build  LLM apps by making Knowledge
#facts, and skills pluggable for LLMs.

# The core usecase of Chroma is to store and retrieve information using embeddings. This core building block
#is at the heart of many powerful AI applications.

# Chroma comes with a built-in embedding model, which makes it simple to load text. We can load the SciQ dataset
#into chroma with just a few lines of code.

# We have to pip install chroma using this pip command (pip install chromadb)

# We can import chromadb like this.

import chromadb

# Next we can create a client using this code.

chroma_client = chromadb.Client()

# Next we can create a collection.

# A collection is where we'll store our embeddings, documents, and any additional metadata. We can create 
#a collection with a name.

# Here is an example of how we can do this.

collection = chroma_client.create_collection(name = "my_collection")

# Next we can add some text documents to our collection.

# Chroma will store our text and handle tokenization, embedding and indexing automatically.

# Here is an example of that.

collection.add(
    documents = ["This is a document", "This is another document"],
    metadatas = [{"source": "my_source"}, {"source": "my_source"}],
    ids = ["id1", "id2"]
) 


# If we already generated embeddings ourselves, we can load them directly in.

# This is an example of how we can do that.

collection.add(
    embeddings = [[1.2, 2.3, 4.5], [6.7, 7.8, 8.9]],
    documents = ["This is a document", "This is another document"],
    metadatas = [{"source": "my_source"}, {"source": "my_source"}],
    ids = ["id1", "id2"]
) 

# Next we will look at Query Collection

# We can query the collection with a list of query text, and chroma will return n most similar results.

# This is an example of what that looks like.

results = collection.query(
    query_texts = ["This is a query document"],
    n_results = 2
)


# By default data stored in Chroma is ephemeral, making it easy to prototype scripts. It's easy to make Chroma
#persistent so we can reuse every collection we create and add more documents to it later. It will load our data
#automatically when we start the client, and save it automatically when we close it. 

# Now we will list some of the differnces between Chroma and Pinecone.

# Pinecone and Chroma are both powerful vector databases, each with its strengths and weaknesses. Pinecone 
#is an excellent choice for real-time search and scalability, while Chroma's open-source nature and flexible 
#querying capabilities make it a versatile option for various applications.

# Here are a few pros and cons between the two.

#       Pinecone

# Pinecone is a managed vector database designed to handle real-time search and similarity matching at scale.
#It is built on state of the art technology and has gained popularity for its ease of use and performance.

# Let's delve into its key attributes, advantages, and limitations:

# We'll start with the Pros

#   Pros
# 1. Real-time search 
# 2. Scalability
# 3. Automatic Indexing
# 4. Python Support

# Now we'll look at the Cons

#   Cons
# 1. Cost
# 2. Limited Querying Functionality


# Now we will be looking at the Pros and Cons of Chroma

#        Chroma

# Chroma, similar to Pinecone, is designed to handle vector storage and retrieval. It offers a robust set of 
#features that cater to various use cases, making it a viable choice for many vector-based apllications.

# Let's delve into its key attributes, advantages, and limitations:

# We'll start with the Pros

#   Pros

# 1. Open-Source
# 2. Extensible Querying
# 3. Community Support

# Now we'll look at the Cons

#   Cons

# 1. Deployment Complexity
# 2. Performance Considerations


# Now we will go over some practical implementation coding using Chroma.

# The first thing we need to do is download our data.

# We can use this link to collect our data by copying it into our browser.

!wget -q https://www.dropbox.com/s/vs6ocyvpzzncvwh/new_articles.zip

# This link will bring us to a dropbox website.

# The data should be a bunch of news articles we can choose from.

# The previous code allowed us to download the srticles to our local machine.

# This code will unzip all of the articles we downloaded to our machince.

!unzip -q new_articles.zip -d new_articles

# More precisely, by using this text data, we will create our chunks, and then we are going to convert those
#chunks into our embeddings, using the embedding model.

# Note, the instructor is using a Jupyter Notebook to write The code we will need to download the articles
#from the website and unzip them. Since we are using VSCode we can not use these steps to follow so we will just
#use our Game Of Thrones PDF again for our data.

# Next we will call the data using the os library. First we have to import it.

import os

# Now that we have imported the os library we can use it to call our data.

# We'll start by importing our openai api key.

# This is how we will do that.

os.environ["OPENAI_API_KEY"] = "sk-proj-NFj08EyWI8laQ9spJBafT3BlbkFJKzklOVqO2yHxrcNMo4lT"

# Now that we have our os library and our import keys, we can import the rest of the libraries we
#need.

# These are the libraries we will be using.

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import openai
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import embeddings
from langchain.chains import RetrievalQA

# Now we will load our data.

# We can do that by implementating a simple code.

# This is the code.

loader = DirectoryLoader("C:/Users/alpha/Generative AI/pdfs/A Game Of Thrones Book 1.pdf")

# Note: The instructor used a glob parameter in his code to access all the articles downloaded from the 
#website using !wget. Since we aren't using those articles we didn't use the glob in our code, but we can always
#read the documentation on glob to get a better understanding of it.

# Basically, glob is for all the text files in the data. It is going to read the data from the entire text files.

# This is how the instructor used glob in his code.

# loader = DirectoryLoader("/home/joyva/work/new_articles", glob = "./*.txt")

# Notice that after glob there is a (./*) The . indicates the current directory, and the * represents all 
#the things in that directory.

# Next we will create a method.

# This is the method we will create.

loader.load()

# Next we will save our load method to a variable called document.

document = loader.load()

# Now we can load or data by calling document.

document

# After getting our data we will create a chunk.

# We can do that by using this code.

# Before doing that we will have to import the text_splitter to our code. See line 177

# Now that we have importred the libraries we need, this is the code we will use to get our chunks.

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
text = text_splitter.split_documents(document)

# Again, let's go over the steps we will be taking.

# First we will get our Data and convert it into our embeddings.

# 1. Data ---------------> Embedding

# The model that we will be using to convert our data into our embedding is the openai embedding model.

# We can a deeper understanding of the openai embedding model by looking at the documentation.

# Next we can say that inbetween our data and our embedding we will have our chunking. That will
#look like this.

# 1. Data ------[Chunking]---------> Embedding

# And after the embedding we will pass everything to our model. That will look like this.

# 1. Data ------[Chunking]---------> Embedding -------> Model

# Let's get an understanding of chunking.

# So let's say we have data that we want to convert into chunks.

# We can perform this task by using this code. 
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Inside this code notice that we two parameters, chunk size and overlap.

# Chunk_size - Specifies that we want to divide our data up into chunks.

# Let's say we want to divide our data into chunks of 100 tokens, for example purposes.

# Let's say we have 3 chunks, for example.

# Each one of these chunks will be 100 tokens long.

# That means that each chunk that we have in our collection of chunks will have a 100 token length.

# Now let's look at the overlap.

# Let's say our overlap is 20

# Overlapping occurs when the last words (how ever many specified by the number we enter) over lap onto the 
#next the chunk.

# Basically, the last 20 words of chunk one will start our chunk two.

# And the same thing goes for the last 20 words of chunk two and the beginning of chunk three.

# Now that we understand that we can move on.

# We can print the variable text to see our chunks.

print(text)

# Now if we wanted to select our first chunk we can use this.

print(text[0])

# This will bring back the first chunk out of the collection of chunks.

# Now we can call the page content to get the contents of our chunk.

print(text[0].page_content)

# Now we can See the contents of our page.

# We can do this for every chunk in our collection by indexing the chunk we want to see.

# We can also see how many chunks we have by running this.

len(text)

# Now that we have all of this, it is time to do the embedding.

# We will start by creating a Data Base.

# The first thing we will do is import the library we need to move forward. This is the library we will
#be using, from langchain import embeddings.

# Now that we have our import, we can start writing the code for our data base.

# This is the code we will be using.

persist_directory = "db"

embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(
    documents = text,
    embedding = embedding,
    persist_directory = persist_directory
)

# Now when we run this code it should generate an embedding in our data base folder.

# There is one disadvantage of the Chroma db.

# Whatever embedding we get back will be in the form of binary.

# After storing our data in the form of embeddings in our db folder, we can mmove on to the next step.

# The next step will be to call the method we created when we stored our data in the db folder.

# This is how we will this.

vectordb.persist()

# This will persist the db to the disk

# Then we will assign it to None

vectordb = None

# Now we can Load the persisted database drom disk, and use it as normal

vectordb = Chroma(persist_directory = persist_directory,
                  embedding_function = embedding) 

# Now if we run our method, we will get our vector data base.

# This is how we will run it.

vectordb


# The next thing we want to do is Make a Retriever.

# To make our retriever we are going to call one more method.

# This is the method we are going to call.

retriever = vectordb.as_retriever() 

# Notice that our new method is the vectordb function as a retriever.

# Also notice that we stored the new method in a variable called retriever.

# Next we will run this particular method by using the retriever.

docs = retriever.get_relevant_documents("How much money did Microsoft raise")

# This is a question that should be able to be answered once our model reads through all of text
#in the articles in our collection.

# We can also take a look inside our docs by writing this command.

docs

# This doc should contain the answer to the question we asked.

# Let's take a look at what is happenning behind the scenes.

# This is the flow of program when we run our docs.

# [Data] ---> [Embeddings] --> [ChromaDB]
#                  |                |
#             [OpenAIAPI]     [Local Disk]
#                          [SQL Lite Server] - Is being used by Local Disk in the backend
#                               [Binary] - How our data gets stored

# Now, when we want to get access to the data we stored in ChromaDB, we will make a retrieval request.

# First we will create a retriever.

# From our created retriever we will make a Query.

# Our query will go directly to the data base where we have our embeddings stored.

# And from this query, we will be returned the output.

#                               Request
# [Retriever] ------> [Query]---------------->
#    | <---------------- | <--------------- | |
#                               Response    | |
#                                           | |
# [Data] ---> [Embeddings] --> [ChromaDB] [DataBase] - This is where we stored our embeddings

# Here we are performing the similarity search.

# And based on the similarity search, we are generating a final output.

# So in our Retriever we are getting a final output.

# We can actually call one more method.

# This is the method we can call.

retriever = vectordb.as_retriever(search_kwargs = {"k": 2})

# This code let's us specify how many answers or responses that we are returned. Here we are specifying
#that we want a max of 2 answers, or responses.

# We would specify this before we run our retriever initially.

# The next thing we will do is make a chain.

# For this we will have to import the library we need. This is the library and import we need to move forward,
#from langchain.chains import RetrievalQA

# Next we will use our llm to call our openai model. This is how we will do that.

llm = openai()

# Now we can call our openai model by using the llm like this.

llm

# Now we will create a chain using this particular method.

qa_chain = RetrievalQA.from_chain_type(llm = openai(),
                                       chain_type = "stuff",
                                       retriever = retriever,
                                       return_source_documents = True)

# So here we have chained together our llm and our retriever inside of our object and stored it inside
#the qa_chain.

# Now we will use a two part method that will be created to cite our sources.

def process_llm_response(llm_response):
    print(llm_response["result"])
    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])

# This is part 1.

# Here is second half of cite method.

query = "How much money did Microsoft raise?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

# This is the second half of our cite method.

# So what we are doing here is creating a method called process_llm_response that will process our queries. 
#Next we created a query that we wanted to request from our model. Next we assigned our Retrievever 
#method (qa_chain) to the variable llm_response.

# Now that our qa_chain(query) is stored in our llm_response, we can call llm_response to get our
#response.

llm_response

# So after all of that, we are not generating an answer directly from our llm (openai) model, we are just
#using it for refinement.

# Instead we will pass our retriever object and generate our answer from there.

# This means we will pass the embeddings, or the database.

# So instead of generating an answer from the model, we will be generating an answer from the 
#embeddings them self.

# Once all of this is done, we can call process_llm_response(llm_response), which will give us our final output
#answer.

process_llm_response(llm_response)

# To recap, the llm_response is not for the answer generation, it returns us an answer from the document 
#itself, and from here we are generating a final answer.

# With all of that being said, let's take a look at the workflow that the project will be following.

# This is the workflow graph that the instructor put together for the project.


#                                                                                                           [USER]
#                                                3                  4                                      /    \ \
#                                                                                                         /      \ \
#                                         [Text Chunk 1] -> [Embeddings 1] --------               [Question]      \ \
#      1                  2              /                                         \                     |         \ \
#                                       /_____[Text Chunk 2] -> [Embeddings 2] -----> [Index]  [Query Embeddings]   \ \
# [PDF Files] -> [Extract Data Contents] _____[Text Chunk 3] -> [Embeddings 3] ------/   |               |           \ \
#                                       \                                           /    |           [Search]         [Llama 2]
#                                        \ -----------------    --------------- ___/     |          /                      /
#                                         [Text Chunk 10] -> [Embeddings 10]___/   [Knowledge Base] ---------------> [Results]
#         
#                                                                                   Vector Store (Chroma DB)

# This is the flow that is happening in our application behind the scene.

