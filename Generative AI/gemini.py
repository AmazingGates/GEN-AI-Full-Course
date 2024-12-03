# In this section we will be going over the concept of Gemnini.

# This is the agenda we will be following in this section of the course.

#   Agenda:

# 1. Google Gemini LLM model - We will be learning what google gemini llm's are, and we will be learning
#about [Muti Model].

# 2. Practical Demo of Google Gemini Pro - We will be seeing and examing google gemini pro llm demos. This
#will help us get familiar with [Text + Vision usecases].

# 3. In the last phase of this section of the module, we will be creating an end to end project using Google
#Gemini Pro.


#   Introduction

# Gemini is built fom the ground up for multimodality - reasoning seamlessly across text, images, videos, audio
#, and code.

# Multimodality: Whenever we talk about multimodality, we are referring to these examples: reasoning seamlessly across 
#text, images, videos, audio, and code.

# The reason why Gemini is considered the most capable AI model is because of its Human Expert(MMLU) score.

# If we do a search we can find out that the MMLU is a (Massive MultiTask Language Understanding) term. 

# Gemini is the first model to outperform human experts on MMLU, one of the most popular methods to test
#the knownledge and problem solving abilities of AI Models.

# Gemini surpasses state-of-the-art performance on a range of benchmarks including text and coding.

# Benchmarking - This falls in the category of reasoning. Here are some examples of benchmarking: 
# :Big-Bench Hard - Diverse set of challenging tasks requiring multi-step reasoning
# :DROP - Reading Comprehension [F1 Score]
# :HellaSwag - Commonsense reasoning for everyday tasks.

# Gemini surpasses state-of-the-art performance on a range of multimodal benchmarks.

# Remember, whenever we are talkng about Multimodals we are talking about data in the form of text, images, 
#videos, and audio.

# Here is a diagram of the multimodal workflow.

#   Gemini: A Family of Highly Capable Multimodal Models
#----------------------------------------------------------

#   Input Sequence
#                                                                                     ________________       ________
# Text----------------------------\                           _______________         | Image Decoder |---->|_Image__|
#                                                            |               |        |_______________|      
# Image---------------------------\                          |               |       /
#                                    ---->[_,_,_,_,_,_]----> |  Transformer  |------{
# Video---------------------------/                          |               |       \________________
#                                                            |_______________|        |               |      ________
# Audio---------------------------/                                                   |_Text Decoder__|---->|_Text___|


# Gemini Supports interleaved sequences of text, image, video, and audio as inputs (illustrated by tokens of different
#colors in the input sequence). It can output responses with interleaved image and text.

# Next we will discuss the different models available to us and their sizes.

# Gemini comes in three sizes.

# 1. Ultra - Our most capable and largest model for high-complex tasks.

# 2. Pro - Our best model for scaling across a wide range of tasks.

# 3. Nano - Our most efficient model for 0n-device tasks.


#   Anything to Anything

# Gemini is natively multimodal, which gives us the potential to transform any type of input into any type
#of output.

# Gemini can generate code based off of different inputs we give it.

# Now that we are more familiar, let's us try some hands on examples.


# This is a link to a tool that we can use.

# ai.google.dev - (https://ai.google.dev/tutorials/python_quickstart?authuser=1)

# This link will take us to a quickstart demostration page.

#   Gemini API: Quickstart with Python

# This quickstart demostrates how to use Python SDK for the Gemini API, which gives us access to Google's
#Gemini Large Language models. In this quickstart we will learn how to:

# 1. Set up our developement environment and API access to use Gemini

# 2. Generate texts responses from text

# 3. Generate texts responses from multimodal inputs (text and image)

# 4. Use Gemini for multiturn conversations (chat)

# 5. Use embeddings for large language models.


#   Prerequisites

# We can run this quickstart in Google Colab, which runs the notebook directly in our browser
#and does not require additional environment configuration.

# With this we will be adding some new elements to our Agenda.


#   ***Agenda Updated***

# 1. Create an API Key 

# 2. Explore multiple examples of Text and Images.

# 3. End to End Project. - Write code from scratch using frontend and backend.


# With that being said, we can now go to our google colab notebook.

# The first thing we will do is visit the link we were giving to access the colab notebook that 
#the instructtor has for us.

# This is the link (https://ai.google.dev/tutorials/python_quickstart?authuser=1).

# Once we use the link and we are inside the google colab notebook we will find the get api key tab and 
#push it.

# This will begin the process os getting our API Key setup for Gemini.

#  If we want to talk to our Model we will need this API Key to make that possible.

# Now let's start working with our notebook.

# The first thing we will do inside our colab note book is install google generative AI package.

# This is the package we will be using.

# !pip install -q -U google-generativeai

# This is the library we will be using.

# It's best that we start a brand new folder for this project, so we'll do that continue there.

# This project folder will be called Gemini, and it will contain our project code and project code
#walkthrough.

# Gemini API Key AIzaSyClGuyL8QtbvDGZvWDraZMLaLD66EIvoJo