from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
import warnings

warnings.filterwarnings("ignore")

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={"temperature": 0})

prompt = "What would be a good name for a company that makes colorful socks?"

print(llm(prompt))


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt_template1 = PromptTemplate(input_variables = ["cuisine"],
                                  template = "I want to open a restaurant for {cuisine} food. Suggest a fancy name.")

input_prompt = prompt_template1.format(cuisine = "Indian")

print(input_prompt)


prompt_template2 = PromptTemplate(input_variables = ["book_name"],
                                  template = "Provide me with a concise summary of the book {book_name}.")

input_prompt = prompt_template2.format(book_name = "Alchemist")

print(input_prompt)


chain = LLMChain(llm = llm, prompt = prompt_template2, verbose = True)

response = chain.run("Game Of Thrones")

print(response)

