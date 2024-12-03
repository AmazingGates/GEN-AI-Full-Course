# In this section we will go over Vector Databases.

# What is a vector database?

# A Vector database is a storing unit for data that is converted into a vector.

# What we learn in this seection:

# :What is a Vector Database?
# :Why we need Vector databases?
# :How Vector databases work?
# :Use cases of Vector databases.
# :Some widely used Vector databases.
# : Practical demo using Python and Langchain


# What is a vector database?

# A vector database is a database used for storing high-dimensional vectors such as word embeddings and 
#image embeddings.

# We can store images or text.

# We are storing in the form of embedding.

# (Things to remember: A Vector is a 2D direction and a Magnitude, which is the distance to a point in that direction.)



# What are embeddings?

# Embeddings are a numerical representation of our data.

# Basically embeddings are vectors, which are a set of numbers, which get displayed in these brackets [ ]

# This bracket will contain the magnitude and the direction, which makes up our vector.

# These are things we can get done with embedding.

# 1. Create Dense Vector
# 2. Context Full (Meaninng Full) Data.

# Let's try to understand embeddings a little better by looking at word 2 vec.

# Let's look an example.

# My name is Brian
# Brian is a Software Engineer
# Brian is working for a start up that specializes in Artificial Intelligence

# The first thing we will do is generate our vocabulary.

# For generating a vocabulary we need to identify our unique words.

# We will only count the first iteration of our words as unique words.

# We have 16 unique words in our three sentences.

# Now we can create our vocabulary.

# [ My name is Brian a Software Engineer working for start up that specializes in Artificial Intelligence ]

# Now let's say we want to create a one hard encoded (OHE) Vector for our first sentence. That would look like this.

# [[ 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ] = My

# [ 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ] = name

# [ 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 ] = is

# [ 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 ]] = Brian


# This is an encoded version of our first sentence which we created using the one hard encoder (OHE)

# We will use this word 2 vect example to explain embeddings.

# So let's say we have some text data.

# From that text data we will create a vocabulary.

# When we're talking about the embedding, we are going to craete some features.

# So we will follow these steps.

# 1. Create Vocabulary
# 2. Create Features from Vocabulary

# This is an example of what the feature and the vocabulary look like.

# This is our example vocabulary

# [ King Queen Man Woman Lion ]

# From this vocabulary we can extract our features.

# This is our feature list.

# 1. Gender
# 2. Wealth
# 3. Power
# 4. Weight
# 5. Speak

# This vocabulary and its Features will be handled by the nueral network.

# The vocabulary will get passed into the N N and the N N will automatically look into the features.

# So we will create our data in such a way that we will have a vocabulary and a N N. 

# We will be passing our vocabulary to the N N and our features will be created.

# And between this we will generate a vector.

# The vector itself will be our embedding.

# This is a high level representation of what our vector will look like.

# Whatever complex mathematics we get from our vocabullary, N N, and features, this is a high level representation
#of that.

# Now let's look at our example vocabuary and feature list again.

# This time we will assign a weight to our vocabulary elements.

# These weights will be between 0-1.

# Now let's add the add values to our elements.

# We will also say that the gender specified is male.

                  # [ King | Queen | Man | Woman | Lion ]
#                          |       |     |       |
# 1. Gender (male)     1   |   0   |  1  |   0   |   1   
# 2. Wealth            1   |   1   | 0.5 |  0.4  |   0
# 3. Power             1   |  0.8  | 0.2 |  0.2  |   0
# 4. Weight           0.8  |  0.6  | 0.7 |  0.5  |  0.6
# 5. Speak             1   |   1   |  1  |   1   |   0

# We have our vectors.

# The King and his values are our first vector.

# And the rest of the vectors follow.

# If we compare this style of vector to our original vector (see line 65), we will see that this style is a dense
#vector and it has more meaning.

# This is a 5D Vector. 

# The way this works is that the weight of features of our 5 vectors are passed into our input layer, then the input
#gets passed into the N N's hidden layers, and the output vector of these are our embeddings.



# Now that we have a simplified understanding of embeddings, the next topic we will look at is why we need vector 
#databases.

#       Why we Need Vector Databases

# Over 80 - 85% of data out there is unstructured data.

# The process of how a vector database is formed looks like this.

#       Doc -  Image  -  PDF    = High Dimensional Data
#                |
#                |
#   [0.34][-0.23][0.99][-0.37]  = High Dimensional Vector
#                |
#                |
#         _______|_______
#        |               |
#        |               |
#        |               |      = Vector Database
#        |               |
#        |_______________|

# Now to breakdown the Vector Database we will first look at the Database.

# There are two forms of databases. The SQL database, where we store our data in the form of rows and columns, and 
#the NoSQL database, where we don't have to write down the SQL or create any kind of schema.

# Also inside the NoSQL database we have different types of databases.

# Next we will look at the Vector, which we went over previoulsy, which will be getting stored inside of our
#database.

# The Vector Database is different from the SQL and the NoSQL databases.

# So let's look into the differences and why we should use this Vector Database.

# First we will look at unstructured data.

# This is basically raw data that is unstructured, like images and videos, where we have pixels in the form of a 
#grid. Pixel values will usually be between 0 - 255. 

# Other forms of unstructured data are text data and voice data.

# These are the most common forms of unstructured data.

# The main problem with unstructured data is that we can't easily store it into a Relational/Traditional database.

# To store something like an image to a traditional database, we would first need to define a schema.

# Here are a few usecases of the Vector Database

# 1. Long-Term memory for LLMs
# 2. Semantic Search: Search based on the meaning of context
# 3. Similarity Search: Text, Images, Videos, Audios
# 4. Recommendation engine as well

# Here are a couple of the most widely used Vector DataBases

# 1. Chroma
# 2. Weaviate
# 3. Pinecone
# 4. FAISS - Scalable Search with Facebook AI

# The first Vector Database we will be looking at is the Pinecone.



#               Now We Will Start Our Practical Implementation Demo.

# The first thing we are going to do is install our libraries.

# The first thing we are going to install is Langchain.

# The thing we are going to install is Pinecone-client.

# The next install is going to be the pypdf

# Next is the pip install openai

# Next is the library tiktoken - This library is important if we are going to call the openai embedding.

# To read over the pinecone documentation and get a deep understanding of it, we can visit pinecone.io

# To have access to pinecone api keys we must sign up in the website. We need the api keys in order to call
#the pinecone api.

# Once all of our libraries are installed we can import them into our code.

# These are the imports we will be using

from langchain.document_loaders import PyPDFDirectoryLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import openai
from langchain.vectorstores import Pinecone
from langchain.chains import retrieval_qa
from langchain.prompts import PromptTemplate
import os
import pinecone
import sys

# The next ting we will do is create a pdf folder using the mkdir in our command terminal. 

# Inside this pdf folder we have to upload the pdf itself, and from there we can upload a text file, or pdf,
#or whatever data we have.

# Next we will get the pdf that we will upload.

# We will go to google and go to the website Attention is all you need research paper.

# Once we open the website, we will download the pdf of the research paper to our local system.

# We will give the file the name Transformers.

# This will be our pdf file.

# If the file is too large we can compress it using a free pdf compressor from google,or use a smaller file 
#on our system.

# Once we upload our pdf to our pdfs file, we can read it into our code.

# This is how we will do that.

PyPDFDirectoryLoader ("pdfs")

# We will assign PyPDFDirectoryLoader ("pdfs") to a variable called loader.

loader = PyPDFDirectoryLoader ("pdfs")

# Then we will call loader.

loader.load

# By using this particular code we willbe able to read our pdf.

# Next we can collect this data which we will be converting into vectors which will make up our embeddings that will
#get stored inside our vector database, which is pinecone.

data = loader.load()

# Now we can see a clean version of our pdf data using this.

data[0]

# The next step is going to be tokenization.

# These are steps we will perform to do that.

# We will call the RecursiveCharacterTextSplitter, and assign it to a variable called text_splitter.

# Then we will pass it a parameter which will be (chunk_size=500, chunk_overlap=20)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)

# Now that we have created an object of it, we can call a method of the object we created.

# This is the method we will call.

# And to this particular method we are passing our data.

text_chunks = text_splitter.split_documents(data)

# Now we can run it to make sure data is coming back in chunks.

text_chunks

# Now we can index into our text_chunks to select specific chunks of data. We can that by doing this.

# This will bring back our first chunk

text_chunks[0]

# We will pass this text_chunks[0] into a print statement to read our chunk of data.

print(text_chunks[0])

# Now we use a page.content on it to get the content of the data we called on our page. We can do that like this.

print(text_chunks[0].page_content)

# We can do the samething for every chunk we have, which should be 500 since we specified it to be 500.

# We can also check the length of each chunk my using this code.

print(len(text_chunks))

# The next thing we will do is get our openai key.

# This is how we will do that.

os.environ["OPENAI_API_KEY"] = "sk-proj-NFj08EyWI8laQ9spJBafT3BlbkFJKzklOVqO2yHxrcNMo4lT"

# Now that we have our chunks we will create an object of our openai embeddings class

# We can go to the openai website to check the documentation on embeddings.

# Once inside the website we will locate and click on the embeddings tab.

# Once inside the embeddings tab we can locate and modify the embeddings code to our specification.

# So if we want to convert our data into a vector, we are going to use the embeddings from the openai.

# This is how we can do that.

# Since we already have the OpenAIEmbeddings imported, we can create and object for it. 

# This is how we will do that.

OpenAIEmbeddings()

# We will save this to a variable called embedding. That will end looking like this.

embedding = OpenAIEmbeddings()

# Next we will call this method, like this.

# And inside this embed query we will pass the question, how are you.

embedding.embed_query("How Are You")

# Once we run this it generate the embeddings.

# It generates the embeddings based on features.

# We can also check to length of the embedding by runnung this code.

len(embedding.embed_query("How Are You"))

# The size of the vector generateed by the embedding will be based on the sentence we passed.

# Now that we have our data and all the configurations we need, we can import the pinecone, and whatever 
#embedding we are going to generate from here.

# The embedding we are going to store in our pinecone database.

# The first thing we will do is write down our pinecone api key.

# There are two variables that will look like this.

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "c1489634-2122-4dc3-a641-e16b3399b247")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV", "gcp-starter")

# Next we will import pinecone(different than the Pinecone import we already have), then we can call the 
#method of it.

# This is how we will do that. (see import implementation on line 254)

pinecone.init(
    api_key = PINECONE_API_KEY,
    environment = PINECONE_API_ENV
)

# We can go to the pinecone documentation to read over everything we have done so far and everything we will
#do concerning the pine cone.

# The next thing we will do is call this init method.

# The next thing we will do is create our index name.

# This is how we will do that.

index_name = "test"

# Next we will go back to the pinecone website and locate indexes.

# Once inside indexes we will click on create new index

# Here we can give our index a name as well as its dimensions.

# Next we will create the index we just named inside of the Pinecone website

# This is how we will do that.

index = Pinecone.Index("testing")

# Next we will create embeddings for each of the text chunks.

# This means that whatever text chunks we have created we will create indexing for.

# This is how we will do that.

docsearch = Pinecone.from_texts([t.page_content for t in text_chunks], embedding, index_name=index_name)

# After we run that particular line of code, we will generate an embedding.

# Now we can go back to the pinecone dashboard and check all the embeddings.

# The embeddings will be in the first row. The text that we tokenized are there also. 

# The embeddings are the values in the first row.

# The text gets converted into embeddings, which get converted into vectors.

# There is also a similarity score.

# This score indicates how similar this sentence is to other sentences.

# We can also access the embeddings by calling the docsearch like this.

docsearch

# Next we will look at an example of a similarity seearch.

# We can do that by performing a query like this first.

query = "A Game Of Thrones Book 1 outperforms which models"

# Next we will find out the similarity search.

# We can do that like this.

docs = docsearch.similarity_search(query)

# Notice we called our similarity search and passed in query as our parameter.

# Now we can run thiss similarity search by calling docs, like this.

docs

# Running it like this will return our similarity search in the form of vectors.

# It is not returned in proper sentences like this.

# Now we will go over the steps of converting the these vectors(numbers) into sentences.

# For this we need to create a LLM.

# This means we need to call the openai api.

# First we'll have our openai method which we will create an object out of.

llm = openai()

# The next thing we'll do is call this particular method, which is called retriever qa.

qa = retrieval_qa.from_chain_type(llm = llm, chain_type = "stuff", retriever = docsearch.as_retriever())

# So this is our class, retrieval_qa, and inside we have a message called from_chain_type().

# from_chain_type() will take the parameters we will specify.

# We store all this in the object qa

# Now we will run it.

# Now we will write down the query again.

query = "A Game Of Thrones Book 1 outperforms which models"

# Next we write qa.run.

# And inside this run method, we are going to pass our query.

qa.run(query)

# This should return a similar similarity search that we have seen regarding this particular question and return 
#to us the answer to the question we asked ("A Game Of Thrones Book 1 outperforms which models").

# Next we will look at how we can create a small qa session.

# So based on the pdf that we are using for the data.

# With that we have generated a vector, then we created the embeddings, and now we will create the query too.

# Also remember that we are getting the similarity search in the form of numbers, which is our vector.

# Once we call the LLM, which will call our openai api.

# We did this by using the retrieval_qa with the from_chain_type method into which we passed our llm, and 
#then we passed the retriever, which is going to be the docsearch itself.

# This is the same docsearch where all of our embeddings are stored.

# The embeddings are stored in this piece of code docsearch.as_retriever().

# Now what we can do is create a small qa session.

# For that we can use this code.

while True:
    user_input = input(f"Input Prompt: ")
    if user_input == "exit":
        print("Exiting")
        sys.exit()
    if user_input == "":
        continue
    result = qa({"query": user_input})
    print(f"Answer: {result["result"]}")

# In this particular piece of code we are saying that if we write down exit, we want to exit the system. But 
#if we don't write down anything, or if we write down anything other than exit, the program will continue.

# The continue will result in our qa which is a user input.

# The user input is an Input Prompt.

# This will lead us to an answer.

# The Answer will be the result of the Input Prompt.

# The next thing we want to is perform the embedding.

# We performed the embedding on top of the data.

# Our data is text data.

# Where did we get this data from?

# We are got the data from our pdf that we used in our pdfs folder.

