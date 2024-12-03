# In This Generative AI Full Course We  Will Be Going Over -
# - Gemini Pro 
# - OpenAI 
# - Llama 
# - Langchain 
# - Pinecone 
# - Vector Databases & More

# sk-proj-NFj08EyWI8laQ9spJBafT3BlbkFJKzklOVqO2yHxrcNMo4lT - OpenAI Key

# 77d19a6c72b61073a6003dd3d80a9f531ec8bf7b1def93e9b12f440fe04fe8ed - serp api key.

# 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 - Hugging Face Key / Read

# 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 - Hugging Face Key / Write

                                                                                                                                                                      






from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(r"C:\Users\alpha\Downloads\Brian J Gates Resume.pdf")

print(loader)

pages = loader.load_and_split()

print(pages)
