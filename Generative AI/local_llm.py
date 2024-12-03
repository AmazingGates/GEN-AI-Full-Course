from langchain_community.llms import HuggingFacePipeline 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain


prompt = PromptTemplate(
    input_variables = ["name"],
    template = "Can you tell me about the YouTuber {name}?"
)


model_id = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map = "auto")

pipeline = pipeline("text2text-generation", model = model, tokenizer = tokenizer, max_length = 128)

local_llm = HuggingFacePipeline(pipeline=pipeline)

chain = LLMChain(llm=local_llm,prompt = prompt)

chain.run("SSSniperwolf")

print(chain.run("SSSniperwolf"))