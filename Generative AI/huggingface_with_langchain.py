# Here we will start using the open source Hugging Face Models.

# The first step will be to import all the libraries that we will be using.

# These are the libraries we will need

# We will be using huggingface with langchain


from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import LLMChain

# Next we will need to set the enviornment.

# We can do this by importing os and then connecting our huggingface key

import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Next we will get the models we want to use.

# Text2Text Generation Models | Seq2Seq Models | Encoder-Decoder Models

# Before we bring over our model we will define our prompt.

# This is how we will define our prompt.

#prompt = PromptTemplate(
#    input_variables = ["product"],
#    template = "What is a good name for a company that makes {product}?",
#    add_to_git_credential = True
#)


# Next we will define our chain.

#chain = LLMChain(llm = HuggingFaceEndpoint(repo_id = "google/flan-t5-large", model_kwargs={"temperature":0}), prompt = prompt)

# Next we will ask our model our question.

# This is how we ask our question to our model.

#chain.invoke("Glasses")




# Note: The code here looks different because our vsc doesn't run depreciated imports so we may end up having a 
#different version here for our vsc.

# Also, the code we are running in the note book is the same as the instructor.

# Next we will use another Text2Text generator known as the facebook/mbart-large-50

# We will implement this model in our notebook right underneath our prompt.

# Then we will take this portion of our chain and replace the google model with the facebook model, which should
#look like this.

# We will also change the temperature of this model to 1.5.

# HuggingFaceHub(repo_id = "facebook/mbart-large-50", model_kwargs={"temperature":1.5})

# Now we can run this model.

# First we need  to create a second chain called chain2, and assign it to our new model, which should look like
#this in our notebook.

# chain2 = LLMChain(llm = HuggingFaceHub(repo_id = "facebook/mbart-large-50", model_kwargs={"temperature":1.5}), propmt = prompt)

# Now we can run the cell to initialize our object.

# Now we can call this new model.

# We will call it like this chain2.run("")

# After running, we discovered that this particular model only returns the prompt with the input we choose, not
#an actual answer.

# Next we will go over the process of using decoder only models.

# The first thing we'll do is import the libraries we need.

# The next thing we'll done is create a prompt.

# We can use the same format as our other prompts.

# This is the prompt we will use.

# These are the libraries we need.

from langchain_community.llms import HuggingFacePipeline 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

#prompt = PromptTemplate(
#    input_variables = ["name"],
#    template = "Can you tell me about the YouTuber {name}?"
#)

# Now we will download our model to our local memory, instead of using the API.

# Now we will go over the steps on how we download these models to use them locally.

# These are the steps we will take to create a pipeline and download our models locally.

# First we will need a model_id, which we will reuse from previously.

#model_id = "google/flan-t5-large"

# Now we need to create the object of our AutoTokenizer and call that particular method.

# This is how we will do that.

#AutoTokenizer.from_pretrained(model_id)

# Now we will store this in a variable, which will now look like this.

#tokenizer = AutoTokenizer.from_pretrained(model_id)

# Now we can run the cell to initialize our object and call this method by passing the model id.

# The next thing we will do is call this method, which is standard procedure for this process.

# This is the model we will call because it is the one we want and notice that it takes as a parameter our 
#model_id.

#model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map = "auto")

# Now we have created our object for this also.

# The next thing we need to do is create our pipeline.

# This is the code we will use to create our pipeline.

#pipeline = pipeline("text2text-generation", model = model, tokenizer = tokenizer, max_length = 128)

# Now we will get our pipeline.

# To get started we have to pass the pipeline we initiated to our huggingface pipeline.

# This is how we will do that by using our local llm.

#local_llm = HuggingFacePipeline(pipeline=pipeline)

# Now that everything is done and clear we can use our local model.

# We will still use the chain to call our model. This is how we will do it.

#chain = LLMChain(llm=local_llm,prompt = prompt)

# Now that we have our chain linked to our local model we can use the chain to ask our question as usual.

#chain.run("SSSniperwolf")

#print(chain.run("SSSniperwolf"))



# In this section we will go over the steps and the process of starting our project.

# We will be discussing the MCQ Generator using OpenAI and Langchain.

# In our first project we will be implementing all the concepts that we have learned so far.

# First we will go over the entire setup of the project.

# Also we won't be using the jupyter notebook to create our project, we will be using a new developer enviornment.


# These are the steps we will take.

# 1. Enviorment setup 

# 2. Run a few experiments in Jupyter Notebook

# 3. Modular Coding

# 4. Create a WebAPI using Steam 

# 5. Deploy Application

# These are the steps we will be using for our first walkthrough project.
