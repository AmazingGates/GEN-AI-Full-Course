# In this section we will be discussing Open Source Large Language Models.

# Open Source LLMs:

# 1. Not hosted anywhere
# 2. Must download the model we want to use
# 3. Must load that model to use it.
# 4. Need good configuration system:
#   : At least core 3 processor
#   : At least 8 gb of RAM
#   : GPU would offer the best experience

# These are the requirements whenever we are talking about the open source llms.

# Here are a few popular and powerful open source llms:

# 1. Meta Llama 2 - This is a Facebook trained Model
# 2. Google PaLM 2 - This is a Google trained Model
# 3. Falcon - Falcon is a Model with many diffrent variants

# There many more open source models but these are the most popular at the time of this course.

# This is what we will be going over in the beginning of this section.

# Introduction to Llama 2

# 1. How to run Llama 2
# 2. How to use Llama 2 with Langchain
# 3. How to build Generative AI project using Llama 2

# We will start by visiting ai.meta.com and following the steps required to download the Llama 2 model.

# This is an extensive step and should done with 24 hours of generating a unique URL from the site.

# We can also visit the Llama2.ai website to start to play around with the model.

# Here are the Variants of Llama 2 Model

# Model Size (Parameters)           Pretrained          Fine-Tuned For Chat Use Cases
#---------------------------------------------------------------------------------------------
#       [7Billion]                   [Model Architecture]   [Data Collection For Helpfulness and Saftey]
#       [13Billion]         [Pre Training Tokens: 2 Trillion]   [Supervised Fine-Tuning: Over 100,000]
#       [70Billion]                [Context Length: 4096]     [Human Prefences: Over 1,000,000]

# We will be using the Llama 2 7B variant.

# We will start learning our model by building with it.

# Here are a few things we should keep in mind whentalking about Data Type and Data Size

# Whenever we're talking about these topics, with these specified bits, we will have this much memory(ram)
#available to us.

#   [Topic]         [32 bits]           [64 bits]

# Character         1 byte              1 byte
# Short/String      2 byte              2 byte
# Integer           4 byte              4 byte
# Long/Float        4 byte              8 byte
# Long Long/Double  8 byte              8 byte

# Notice that floats take up more space in the memory 

# We will Start to get familiar with the Llama model using the Llama 13 B in a Jupyter Notebook.

# We will start by getting these installs.

!CMAKE_ARGS = "-DLLAMA_CUBLAS=on" FORCE_CMAKE = 1 pip install llama-cpp-python==0.1.78  numpy==1.23.4 --force-reinstall --upgrade --upgrade --no-cache-dir --verbose
!pip install huggingface_hub
!pip install llama-ccp-python==0.1.78
!pip install numpy==1.23.4

# The reason we are using the Jupyter Notebook is because we have getting specific versions of the libraries we need,
#and to avoid the installs we already have on our loca machince, we can just run this project in a Notebook.

# The first thing we will do is write down our Models name and path

# This is how we will do that

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # The model is in bin format

# Next we will get the imports we need.

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Now we will download our model.

# This is how we will do that.

model_path = hf_hub_download(repo_id = model_name_or_path, filename = model_basename) 

# Now we will run the cell with the hf_hub_download code to download our model.

# When the model is done downloading and we want to check the model path, we can just run command in 
#our notebook

model_path

# The next thing to do will be to load our model.

# We need to use the Llama cpp library to load our model.

# This is how we will use the llama cpp to load our model

lcpp_llm = None
lcpp_llm = Llama(
    model_path = model_path,
    n_threads = 2, # CPU Cores
    n_batch = 512, 
    n_gpu_layers = 32
)

# Now that we have this code will run this cell in the notebook to load our Model.

# When the model is complete we will create our first prompt template.

# This is how we will do that.

prompt = "Write a linear regression code"
prompt_template = f'''SYSTEM: You are a helpful, respectful and honest assistant. You always answer with precision.

User: {prompt}

ASSISTANT:
'''

# Now that we have our prompt we will execute and give it to our model/llm.

# This is how we will do that.

response = lcpp_llm(prompt, prompt_template, max_tokens = 256, temperature = 0.5, top_p = 0.95,
                    repeat_penalty = 1.2, top_k = 150,
                    echo = True)

# We will also set our parameters. max_tokens (how many many words max will be in our reponse), and the
#temperature(how creative our responses will be.)

# If we are unable to run the code in our local machine or jupyter notebook, we can also try google colab.

# If everything is running smooth, the next command we can print is the response.

print(response)

# Next we simplify our response get back a cleaner version by doing this.

print(response["choices"][0]["text"])

# This will show us the actual response of the assistant.

# This code and assistant is now complete.
