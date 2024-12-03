# In this section we will be going over the process of createing a Help Management App.

# Here is the agenda for this project.

#       Agenda

# 1. Understanding what is Generative AI.
# 2. How LLM models are trained.
# 3. Open Source and Paid LLM's.
# 4. Explore Famous Models - Example: GPT-4, Llama 2, Gemini Pro, etc..

# These are the topics we will be discussing step by step.

# Let's start by looking at Generative AI.

# But before we start looking at Gen AI, let's think about where it fits in the universe 
#of Artificial Intelligence.

# Let's look at the graph below for references.


#        ____________________Artificial Intelligence_______
#       |                                                  |
#       |    _____________                                 |
#       |   |  _________  |                                |
#       |   | |         | |                                |
#       |   | | [GenAI] | |                                |
#       |   | |_________| |                                |
#       |   |__|__________|                                | 
#       |______|___|_______________________________________|
#              |   |
#              |   Machine Learning --> Stats tool --> Analyze Data, Models, Forecasting, Predictions
#              |
#              Deep Learning --> Multi Layered Neural Networks


# Here are a few more techniques of Deep Learning that is responsible for "solving some specific usecases"

# ANN
# CNN
# RNN

# Now that we have an understanding of this, we can move forward to see exactly where Generative AI fits
#into this universe.

#   Generative AI:

# Generative AI is considered a subset of Deep Learning.

# Why it falls under the subset of Deep Learning is because we are still using Deep Learning techniques
#here.

# Most of the Models we will see inside Generative AI are based on the [Transformers, BERT] Models.

# Inside Generative AI we have LLM's (Large Language Models) and LIM's (Large Image Models).

# LLM's will handle all of the issues pertaining to Text.

# LIM's will handle all of the issues pertaining to Images.

# Note: There are Models out there that we have already used that can handle both of these functions.

# Gemini Pro handle can both, issues pertaining to Text, and handle issues pertaining to Images.

# This is why Gemini is considered Multi-Modal.


# Now that we are able to understand what Generative AI is, we will start by discussing the two 
#main topics inside genai, which are LLM's and LIM's.

# First of all, the question we should be asking is what exactly is Generative AI.

# Some years back, traditional learning algorithms were used.

# First there was Traditional Machine Learning Algorithms.

# Traditional ML Algorithms - Feature Engineering, Model Training, Fine Tuning, Deployment.

# Then we moved towards Traditional Deep Learning Algorithms.

# Traditional DL Algorithms - The best way to describe why we moved towards the DL algorithms
#is to look at the example graph below.

# Machine Learning Graph

# Performance
#   |
#   |
#   |
#   |
#   |
#   |_________________
#                 Dataset

# With traditional machine learning algorithms we would see that when data was increasing after 
#one point of time, the performance of the trad ml algorithm would start plateauing.

# Performance
#   |
#   |
#   |      ______________
#   |    /
#   |  /
#   |/_________________
#                 Dataset

# Even though we may increase the data, after a certain point in time plateauing will occur.

# This is the issue that brought about Deep Learning Algorithms.

# With deep learning (specifically multi layered neural networks), as we start increasing the dataset
#the performance increases also.

# This is the reason Deep Learning Algorithms became famous.

# Now deep learning algorithms are used for various usecases.

# Traditional Deep Learning Algorithms - Supervised Learning, Unsupervised Learning, Computer Vision,
#NLP, etc..

# There are many task and usecases that Deep Learning Algorithms are used to solve.

# For a deeper understanding, let's look at the graph below.

#           Deep Learning
#                |
# _______________|_________________
# |                               |
# Discriminative                  Generative  
#      |                               |
#    Tasks                           Tasks-- Generate New Data Trained On Some Dataset.
# -----------------------
#  |          |         |
# Classify   Predict   Object Detection (Any supervised or Unsupervised Techniques) 
#                                                   |
#                                   These models are usually trained on labeled Datasets  
#                                                   |
#                 These models fall under the branch of Traditional Deep Learning Algorithms
#   

# This graph helps us visualize the differences between the different model types inside of 
#deep learning.

# These are the Discriminative Models and the Generative Models.

# Basically, all of the deep learning algorithms are based on one of these two Model Types.

# On the other hand, we can notice that the main function of the Generative Models is to generate 
#new data that has been trained on some dataset.

# Example: Write an essay on generative ai.

# This is the kind of task we can assign to the generative ai algorithm models.

# Now assuming that the model was trained on huge amounts of datasets that pertain to the queries
#we will be making, we should expect a response to the query we asked our model about.

# For a deeper understanding of What a generative ai task looks like, let's look at the real world 
#example below.


#                        Senior   MCAT
# Generative AI ------> [Person] ------> [Doctor] --> Medical Chatbot
#                                 4 + 1


# Now let's break down this graph for deeper understanding.

# Notice that first we have our model.

# Then our model generates a Person. This person is a senior in college.

# The Person is studying for the biggest test of their career field, the MCAT.

# The Person has spent four years in school, and 1 year interning.

# The final step of this process is for the student to become a Docotr.

# This Doctor, will be the model beind our queries.

# Now we have a Medical Chatbot Doctor, who we presume has been trained on large amounts 
#of Medical Data.

# Now we can query our model and with that, visualize another addition to our code flow.

# This is what the codes flow looks like now.



#                                           |--> LLM --> Medicine --> [2 - 3] --> Books
#                        Senior   MCAT      |
# Generative AI ------> [Person] ------> [Doctor] --> Medical Chatbot
#                                 4 + 1     |
#                                           |
#                                        Question --> Generic Medical Problem
#                                           |
#                                           |
#                                        Response 


# Now, we can see that when we go and ask the Doctor any question, related to any generic medical 
#problem, we can expect a response from our model.

# Now our model is like a LLM, which has been trained on large amounts of data, to provide
#generated responses for queries from all of the data that the model has been trained on.

# This makes our LLM an expert in Medicine.

# This is exactly what Open AI is trying to do.

# Open AI is planning to come with something called GPT store.

# Basically this GPT Store means that we now have the ability to create our own models, and train
#it with our own custom data.


# Open AI --> GPT STORE
#                 |
#          LLM Models --> Custom Data


# Now back to our LLM, which is an expert in medicine.

# At this point the next step for our LLM/Doctor would be to become an MD.

# There are some questions which our doctor will not understand, which means our Doctor may
#not be able to answer the question correctly. It may give us a generic answer based on how
#it understands the data and the query. 

# This means we need to train our LLM Model again with more data.

# This time we want train with special editions, which will be what takes our LLM/Doctor from
#expert in medicine to an MD (of a particular field).

# This LLM/MD will have been trained for another 2 to 3 more years using specialized data from
#books of the particular field it happens to be in.

# This, along with experience is what makes our LLM grow from a Doctor, to an MD.

# This means our LLM/MD is able to generate its own responses based on the problem statement it sees.

# Now we can move on to the second step, which is usecases.

# We will be discussing usecases with respect to both techniques.

#       Usecases
#   ---------------

# Discriminative - Discriminates based on the data and generates some sort of output.

# Let's lookk at an example below.

#    ___________
#   |           |
#   |   Types   |
#   |   of      |
#   |   Music   |
#   |___________|

# We can say that this our dataset which we named types of music.

# From this we will want to create a discriminative model. 

#    ___________        ________________
#   |           |      |                |
#   |   Types   |      | Discriminative |
#   |   of      | ---> | Deep Learning  |
#   |   Music   |      |    Model       |
#   |___________|      |________________|

# This Model will be our classifier that helps put the different genres of music in their 
#respective category.

# There will be three categories to choose from, Classical, Rock or Rap.

#    ___________        ________________       Classial
#   |           |      |                |    /
#   |   Types   |      | Discriminative |  /
#   |   of      | ---> | Deep Learning  |/------- Rock
#   |   Music   |      |    Model       |\
#   |___________|      |________________|  \
#                                            \
#                                              Rap

# This is basically the discriminative technique.


# Now let's move on to the next one, which is known as the Generative Technique.

# Generative - 

# Again, let's say we have music for this example.

#    ______________
#   |              |
#   |      ___    /|
#   |     /  /_  / |
#   | _/\/    /_/  |
#   |/_____________|

# From this music we will train our Generative Model.

#    ______________       ____________
#   |              |     |            |
#   |      ___    /|     | Generative |
#   |     /  /_  / | --> |            |
#   | _/\/    /_/  |     |   Model    | 
#   |/_____________|     |____________|

# The task of this generative model will be to generate new music from the music we provided.

#    ______________       ____________
#   |              |     |            |       ___________
#   |      ___    /|     | Generative |      |           |
#   |     /  /_  / | --> |            |  --> | New Music |
#   | _/\/    /_/  |     |   Model    |      |___________|
#   |/_____________|     |____________|

# This is just one usecase of Generative AI.

# In short, this usecase will allow us to generate new content from the data we entered.

#    ______________       ____________
#   |              |     |            |       ___________
#   |      ___    /|     | Generative |      |           |
#   |     /  /_  / | --> |            |  --> | New Music | --> New Content
#   | _/\/    /_/  |     |   Model    |      |___________|
#   |/_____________|     |____________|


# Now we will look at the main question.

# How LLM Models are trained.

# According to research papers, the process goes as such.

# The first section of training reviewed in the paper is "Pretraining".

# The main topics of pretraining consist of, Pretraining Data, Training Details, and
#Llama 2 Pretrained Model Evaluation.

# The next section of discussion is "Fine-Tuning"(SFT).

# The main topics of fine-tuning consist of, Supervised Fine Tuning, Reinforcement Learning with
#Human FeedBack(RLHF), System Message for Multi-Turn Consistency, and finally RLHF Results.

# There are more sections of the paper but the first two will be the focus.

# This is the process for training of a Llama 2 Chat Model.

# This process begins with the pretraining of Llama 2 using publicly available online resources.

# Following this, we create an initial version of Llama 2 Chat through the application of 
#supervised fine-tuning.

# Subsequently, the model is iteratively refined using Reinforcement Learning with Human Feedback
#(RLHF) methodologies, specifically through rejection sampling and Proximal Policy Optimization(PPO).

# Throughout the RLHF stage, the accumulation of iterative reward modeling data in parallel with 
#model enhancements is crucial to ensure the reward models remain within distribution.

# Here is a visual example of this in graph form below.

#   Human Feedback           _____________________________________________________________________________________
#                           |                                                                                     |
#                           |              /-----> Safety Reward Model ----------                                 |          
#                Human Performance Data ---                                      \___________________             |                    
#                                          \-----> Helpful Reward Model ---------/                   |            |                                               
#                                                                                            Fine-Tuning          |
#                                                                                                                 |
#                                                                                Rejection Sampling <---> (PPO)   |
#                                                                                         \                /      |
#                                                                                          \    RLHF      /       |
#                                                                                           \            /        |
#   Pretraining Data --> Self Supervised Learning --> Llama 2 --> Supervised Fine-Tuning --> Llama 2 Chat---------|


# Now we will take our time and going each step in this process.

#   Pretraining

# To create the new family of Llama 2 models, we began by using an optimized auto-regressive transformer,
#but made several changes to improve performance.

# Pretraining Data - Our training corpus includes a new mix of data from publicly sources, which does not
#incluce data from Meta's products or services.

# Training Details - We adopt most of the pretraining setting and model archictecture from Llama 1.

# Hyperparameters - We trained using the Adam W optimizer.


# Now let's get a deeper understanding on how these models are trained.

# We'll start with the stages of training.

#   Stages Of Training:

#       Stage 1                                     Stage 2                       Stage 3
#    __________________________                ____________________          _______________
#   |                          |              |   Supervised Fine  |        | Reinforcement |    
#   | Generative Pre Training  |------------->|     Tuning(SFT)    |------->| Through Human |
#   |__________________________|              |____________________|        |____Feedback___|


# Stage 1 - Initially in this stage we get our data. Which can be any LLM Model, for example,
# INTERNET TEXT DATA or DOCUMENT TEXT DATA.

# Here, we specifically use the Generative Pre Training.

# What that means is that we'll basically use a TRANSFORMER ARCHITECTURE MODEL

# The outcome of this is the BASE TRANSFORMER MODEL.

# This BASE TRANSFORMER MODEL is basically whatever transformer we trained on.

# That model then gets connected to SUPERVISED FINE TUNING(SFT).

# This BAS TRANSFORMER MODEL will be able to do various task, like text classification,
#text summarization, etc...


# Now we will look at Stage 2

# Stage 2 - Here, what we are specifically going to do, we're going to use Human Trainers also.

# We will rely on human Trainers to create some kind of conversation.

# It is also important to understand that here, we will create some custom data.

# Basically, this is the step where custom data is created.

# This custom data is created by Human Trainers.

# Based on this custom data, we will train this specific model, and the outcome of this model will be
#coneecttted to the final stage in this process, which is where REINFORCEMENT LEARNING THROUGH HUMAN
#FEEDBACK happens.
