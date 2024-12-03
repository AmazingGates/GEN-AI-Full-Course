# Now we will be moving forward using Llama with Langchain. We will be using the Llama-7B with Langchain.

# The first thing we will do is get our installs

# These are the install we need [ transaformers, einops, accelerate, bitsandbytes ]

# With our installs installed, we can start on our imports.

from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
import warnings

# Now that we have our imports we can start writing our code.

warnings.filterwarnings("ignore")

# The next thing we will do is load our Llama2_7B model.

# We will use two pieces of code next. The first code is if we have access to the hugging face version
#of Llama2 7B.

# The second version is a cloned repository created by a user.

# Note: If we are using one the other must be commented out.

# We will be using the first version.

model = "meta-llama/Llama-2-7b-chat-hf"
#model = "dary1149/llama-2-7b-chat-hf"

# Now we can load our tokenizer

tokenizer = AutoTokenizer.from_pretrained(model)

# Next we will create our pipeline.

pipeline = transformers.pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    torch_dtype = torch.bfloat16,
    trust_remote_code = True,
    device_map = "auto",
    max_length = 1000,
    do_sample = True,
    top_k = 10,
    num_return_sequences = 1,
    eos_token_id = tokenizer.eos_token_id
)

# How will this hugginface pipeline operate?

# Input Text ------> [Pipeline] ------> Response

# So what is happenning inside our Pipeline:

# 1. First we will apply some pre processing, with the help of the Auto Tokenizer

# 2. Then we will convert the Text to Numbers(Vector)

# 3. Then the Vector will get passed to our model, where the prediction will happen.

# 4. Then we will get our response.

# This is the Pipeline flowchart of what's taking place inside our Pipeline when we run our code.

llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {"temperature:0"})

prompt = "What would be a good name for a company that makes colorful socks?"

print(llm(prompt))


# Next we will go over prompt templates.

# Currently in the above applications we are writing an entire prompt, but if we are creating a user directed 
#application, then this is not an ideal case.

# Langchain facilitates prompt management and optimization.

# Normally when we use an LLM in an application, we are not sending user input directly to the LLM. Instead,
#we need to take the user input and construct a prompt, and only then send it to the LLM.

# In many Large Language Model applications we do not pass the user input directly to the LLM, we add the
#user to a large piece of text called prompt template.

# This was brief overview of the prompt template.

# Now we will get the imports we need to move forward.

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Now we will write our first example.

prompt_template1 = PromptTemplate(input_variables = ["cuisine"],
                                  template = "I want to open a restaurant for {cuisine} food. Suggest a fancy name.")

input_prompt = prompt_template1.format(cuisine = "Indian")

print(input_prompt)


# Now we will look at example 2

prompt_template2 = PromptTemplate(input_variables = ["book_name"],
                                  template = "Provide me with a concise summary of the book {book_name}.")

input_prompt = prompt_template2.format(book_name = "Game Of Thrones")

print(input_prompt)


# Now we will execute our final chain.

chain = LLMChain(llm = llm, prompt = prompt_template1, verbose = True)

response = chain.run("Indian")

print(response)


chain = LLMChain(llm = llm, prompt = prompt_template2, verbose = True)

response = chain.run("Game Of Thrones")

print(response)


# We can also try to use different variants of the Llama2 model since other variants have different functions
#and usecases.

# We can explore HuggingFace to get more understanding and practice with the Models.


